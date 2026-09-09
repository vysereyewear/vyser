// Sincroniza o catálogo da loja Shopify para public/catalog.json
//
// Usa o endpoint público /products.json — não precisa de token nem de app privado.
//
// A loja é organizada com UM PRODUTO POR COR ("BLAZE - BLACK / BLACK"), sem
// featured_image nas variantes. Então frente e ladinho saem da numeração no nome
// do arquivo: dentro de um produto, ordenando pelo número, as fotos seguem sempre
// a mesma sequência de captação. Para óculos e bonés é [ladinho, frente, lado,
// costas/modelo...]; para joias, a primeira já é a foto cheia.
//
// Fotos com modelo e banners entram no meio da numeração, então cada candidata é
// conferida pela fração de fundo antes de ser aceita.

import fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const STORE = process.env.SHOPIFY_STORE_DOMAIN || 'vyser-eyewear.com';
const OUT = path.join(__dirname, 'public', 'catalog.json');

// product_type da loja → aba do app
const GRUPO_POR_TIPO = {
  sunglasses: 'oculos',
  cap: 'bone',
};

function grupoDe(productType) {
  const t = (productType || '').trim().toLowerCase();
  return GRUPO_POR_TIPO[t] || 'joia';
}

// "BLAZE - BLACK / BLACK" → { base: 'BLAZE', cor: 'BLACK / BLACK' }
// Sem hífen, o produto é uma cor só e vira o próprio nome.
function separarTitulo(titulo) {
  const i = titulo.indexOf(' - ');
  return i === -1
    ? { base: titulo.trim(), cor: 'Único' }
    : { base: titulo.slice(0, i).trim(), cor: titulo.slice(i + 3).trim() };
}

// número no começo do nome do arquivo: "12_a1b2c3.jpg" ou "12.jpg" → 12
function numeroDoArquivo(src) {
  const nome = src.split('/').pop().split('?')[0];
  const m = nome.match(/^(\d+)[_.]/);
  return m ? parseInt(m[1], 10) : null;
}

// Foto de produto é objeto pequeno sobre fundo liso, então quase todo o quadro é
// fundo. Foto com modelo tem uma pessoa ocupando o quadro e derruba essa fração.
// Medido no catálogo: produto fica em 68-89%, modelo em 3-59%.
const FRACAO_MINIMA_DE_FUNDO = 0.64;
const cacheFundo = new Map();

async function fracaoDeFundo(url) {
  if (cacheFundo.has(url)) return cacheFundo.get(url);

  let fracao = null;
  try {
    const res = await fetch(url.split('?')[0] + '?width=64');
    if (res.ok) {
      const buf = Buffer.from(await res.arrayBuffer());
      const { data, info } = await sharp(buf)
        .resize(64, 64, { fit: 'fill' }).removeAlpha().raw()
        .toBuffer({ resolveWithObject: true });

      const px = (x, y) => {
        const i = (y * info.width + x) * info.channels;
        return [data[i], data[i + 1], data[i + 2]];
      };
      const cantos = [px(1, 1), px(62, 1), px(1, 62), px(62, 62)];
      const fundo = [0, 1, 2].map((c) => cantos.map((p) => p[c]).sort((a, b) => a - b)[1]);

      let n = 0;
      for (let i = 0; i < data.length; i += info.channels) {
        const dist = Math.abs(data[i] - fundo[0]) + Math.abs(data[i + 1] - fundo[1]) + Math.abs(data[i + 2] - fundo[2]);
        if (dist < 30) n++;
      }
      fracao = n / (info.width * info.height);
    }
  } catch {
    fracao = null; // sem rede ou imagem quebrada: fica "não sei"
  }

  cacheFundo.set(url, fracao);
  return fracao;
}

// As fotos só de produto, na ordem de captação, checando no máximo `limite`
// candidatas pra não baixar o produto inteiro.
async function fotosDeProduto(images, limite = 4) {
  const numeradas = images
    .map((img) => ({ src: img.src, n: numeroDoArquivo(img.src) }))
    .filter((x) => x.n !== null)
    .sort((a, b) => a.n - b.n);

  const aceitas = [];
  for (const foto of numeradas.slice(0, limite)) {
    const fracao = await fracaoDeFundo(foto.src);
    if (fracao === null || fracao >= FRACAO_MINIMA_DE_FUNDO) aceitas.push(foto.src);
    if (aceitas.length === 2) break;
  }
  return aceitas;
}

export async function sincronizarCatalogo() {
  const res = await fetch(`https://${STORE}/products.json?limit=250`);
  if (!res.ok) throw new Error(`Shopify respondeu ${res.status} ao buscar /products.json`);
  const { products } = await res.json();

  const porBase = new Map();

  for (const p of products) {
    // Kits não são um produto pra vestir no modelo — a foto é da caixa/sacola
    if (/\bkit\b/i.test(p.title) || /^kit$/i.test((p.product_type || '').trim())) continue;
    if (!p.images?.length) continue;

    const grupo = grupoDe(p.product_type);
    const { base, cor } = separarTitulo(p.title);

    const fotos = await fotosDeProduto(p.images);
    if (!fotos.length) continue;

    // Em óculos e bonés a primeira da sequência é o 3/4 e a segunda é a frente.
    // Em joias a primeira já é a foto cheia do produto.
    const [frente, ladinho] = grupo === 'joia'
      ? [fotos[0], fotos[1] || null]
      : [fotos[1] || fotos[0], fotos[1] ? fotos[0] : null];

    if (!porBase.has(base)) {
      porBase.set(base, { id: String(p.id), titulo: base, handle: p.handle, grupo, cores: [], todas: [] });
    }
    const entrada = porBase.get(base);
    entrada.cores.push({ id: String(p.id), nome: cor, frente, ladinho });
    entrada.todas.push(...p.images.map((img) => img.src));
  }

  const dados = {
    loja: STORE,
    sincronizadoEm: new Date().toISOString(),
    produtos: [...porBase.values()].sort((a, b) => a.titulo.localeCompare(b.titulo)),
  };

  await fs.writeFile(OUT, JSON.stringify(dados, null, 2), 'utf8');
  return dados;
}

// Rodando direto: node sync-catalog.mjs
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const dados = await sincronizarCatalogo();
  const porGrupo = dados.produtos.reduce((acc, p) => {
    acc[p.grupo] = (acc[p.grupo] || 0) + 1;
    return acc;
  }, {});
  const cores = dados.produtos.reduce((n, p) => n + p.cores.length, 0);
  const semLadinho = dados.produtos.flatMap((p) => p.cores.filter((c) => !c.ladinho));
  console.log(`catalog.json escrito: ${dados.produtos.length} produtos (${JSON.stringify(porGrupo)}), ${cores} cores`);
  if (semLadinho.length) console.log(`sem ladinho: ${semLadinho.length}`);
}
