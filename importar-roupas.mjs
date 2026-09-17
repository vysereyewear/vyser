// Importa um export do Notion (ou qualquer pasta de imagens) para o guarda-roupa.
//
//   node importar-roupas.mjs <pasta>
//
// Cada imagem é olhada por um modelo de visão, que diz se é peça de cima ou de
// baixo e dá um nome descritivo — o nome do arquivo é o que vira o rótulo na
// grade do app e a descrição no prompt, então nome ruim estraga a geração.

import 'dotenv/config';
import fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODELO = 'gpt-5.4-mini';
const EXTS = ['.png', '.jpg', '.jpeg', '.webp'];

const entrada = process.argv[2];
if (!entrada) {
  console.error('uso: node importar-roupas.mjs <pasta com as imagens>');
  process.exit(1);
}

const INSTRUCAO = `This is a photo from a streetwear brand's reference folder.

Answer as JSON:
{"tipo":"cima"|"baixo"|"outro","nome":"<short kebab-case name in Portuguese>"}

- "cima": anything worn on the upper body — jacket, hoodie, sweatshirt, coat, vest, shirt, t-shirt.
- "baixo": anything worn on the lower body — trousers, jeans, cargo, shorts, sweatpants, skirt.
- "outro": anything that is not a garment (sunglasses, shoes, bags, jewelry, a person wearing a full outfit, a logo, a mockup).

The "nome" describes the garment so someone reading only the name can picture it: type, colour and a distinctive detail. Examples: "moletom-rosa-destroyed", "jaqueta-couro-preta", "calca-cargo-bege", "shorts-jeans-lavado". Lowercase, hyphens, no accents, at most 5 words.`;

async function classificar(arquivo) {
  const b64 = (await sharp(arquivo).resize(512, 512, { fit: 'inside' }).jpeg({ quality: 80 }).toBuffer()).toString('base64');
  const r = await client.chat.completions.create({
    model: MODELO,
    messages: [{ role: 'user', content: [
      { type: 'text', text: INSTRUCAO },
      { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${b64}` } },
    ]}],
    response_format: { type: 'json_object' },
  });
  return JSON.parse(r.choices[0].message.content);
}

function semColidir(usados, nome, ext) {
  let final = nome + ext, n = 2;
  while (usados.has(final)) final = `${nome}-${n++}${ext}`;
  usados.add(final);
  return final;
}

const arquivos = (await fs.readdir(entrada))
  .filter(f => EXTS.includes(path.extname(f).toLowerCase()))
  .sort();

const destinos = {
  cima: path.join(process.cwd(), 'public', 'roupas'),
  baixo: path.join(process.cwd(), 'public', 'calcas'),
};
await fs.mkdir(destinos.cima, { recursive: true });
await fs.mkdir(destinos.baixo, { recursive: true });

const usados = new Map([['cima', new Set(await fs.readdir(destinos.cima))], ['baixo', new Set(await fs.readdir(destinos.baixo))]]);
const resultado = [];

// 4 por vez: rápido sem estourar a API
let proxima = 0;
const trabalhador = async () => {
  while (proxima < arquivos.length) {
    const f = arquivos[proxima++];
    try {
      const { tipo, nome } = await classificar(path.join(entrada, f));
      if (tipo !== 'cima' && tipo !== 'baixo') {
        resultado.push({ origem: f, tipo: 'outro' });
        continue;
      }
      const ext = path.extname(f).toLowerCase();
      const destino = semColidir(usados.get(tipo), nome, ext);
      await fs.copyFile(path.join(entrada, f), path.join(destinos[tipo], destino));
      resultado.push({ origem: f, tipo, destino });
    } catch (e) {
      resultado.push({ origem: f, tipo: 'erro', erro: e.message });
    }
  }
};
await Promise.all(Array.from({ length: 4 }, trabalhador));

resultado.sort((a, b) => a.origem.localeCompare(b.origem));
for (const r of resultado) {
  console.log(`${r.origem.padEnd(30)} ${r.tipo.padEnd(6)} ${r.destino || r.erro || ''}`);
}
const conta = t => resultado.filter(r => r.tipo === t).length;
console.log(`\nroupas: ${conta('cima')} | calcas: ${conta('baixo')} | ignoradas: ${conta('outro')} | erros: ${conta('erro')}`);
