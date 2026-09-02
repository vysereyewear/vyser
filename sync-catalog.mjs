// Sincroniza o catálogo da loja Shopify para public/catalog.json
//
// Usa o endpoint público /products.json — não precisa de token nem de app privado.
// Cada variante de cor tem uma featured_image no Shopify, e essa é sempre a foto
// de FRENTE. As fotos de cada cor são numeradas em sequência no nome do arquivo
// (1_hash.jpg, 2_hash.jpg, ...), então o "ladinho" (3/4, não completamente de lado)
// é o arquivo N-1 — com fallback pro vizinho seguinte quando N-1 não existe.

import fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const STORE = process.env.SHOPIFY_STORE_DOMAIN || 'vyser-eyewear.com';
const OUT = path.join(__dirname, 'public', 'catalog.json');

// product_type do Shopify → grupo usado nas abas do app.
// Óculos são os que estão sem product_type na loja, então o título desempata
// os que ficaram sem tipo preenchido (ex: TALON CAP).
function grupoDe(productType, titulo = '') {
  const t = (productType || '').toLowerCase();
  if (t.includes('cap')) return 'bone';
  if (t === '') return /\bcap\b/i.test(titulo) ? 'bone' : 'oculos';
  return 'joia';
}

// Dentro de joia, o tipo decide onde a peça é usada (dedo, pulso, pescoço)
function subtipoDe(productType, titulo = '') {
  const t = `${productType} ${titulo}`.toLowerCase();
  if (/\bring\b|\banel\b/.test(t)) return 'anel';
  if (/bracelet|armband|pulseira/.test(t)) return 'pulseira';
  if (/chain|necklace|halskette|colar/.test(t)) return 'corrente';
  return null;
}

// número no começo do nome do arquivo: "12_a1b2c3.jpg" ou "12.jpg" → 12
function numeroDoArquivo(src) {
  const nome = src.split('/').pop().split('?')[0];
  const m = nome.match(/^(\d+)[_.]/);
  return m ? parseInt(m[1], 10) : null;
}

// Foto de produto é um objeto pequeno sobre fundo liso, então quase todo o quadro
// é fundo. Foto com modelo tem uma pessoa ocupando o quadro e derruba essa fração.
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
      // cor do fundo = mediana dos quatro cantos
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

// A foto de frente é a N do lote da cor, e o ladinho (3/4) costuma ser a N-1.
// Mas quando a cor começa no próprio N, a N-1 é do lote anterior e cai justo numa
// foto com modelo — por isso cada candidata é conferida antes de ser aceita.
async function escolherLadinho(images, idxFrente, frentesDeOutrasCores) {
  const n = numeroDoArquivo(images[idxFrente].src);

  const maisProximaComNumero = (alvo) =>
    images
      .map((img, i) => ({ img, dist: Math.abs(i - idxFrente) }))
      .filter(({ img, dist }) => dist > 0 && numeroDoArquivo(img.src) === alvo)
      .sort((a, b) => a.dist - b.dist)[0]?.img.src || null;

  // N-1 primeiro (o 3/4 na maioria dos lotes), depois os vizinhos seguintes
  const candidatas = (n === null
    ? [images[idxFrente + 1]?.src, images[idxFrente - 1]?.src]
    : [n - 1, n + 1, n + 2, n - 2].map(maisProximaComNumero)
  ).filter((src) => src && !frentesDeOutrasCores.has(src));

  let primeiraDesconhecida = null;
  for (const src of candidatas) {
    const fracao = await fracaoDeFundo(src);
    if (fracao === null) {
      primeiraDesconhecida ??= src; // não deu pra medir; só usa se nada melhor aparecer
    } else if (fracao >= FRACAO_MINIMA_DE_FUNDO) {
      return src;
    }
  }
  return primeiraDesconhecida;
}

export async function sincronizarCatalogo() {
  const res = await fetch(`https://${STORE}/products.json?limit=250`);
  if (!res.ok) throw new Error(`Shopify respondeu ${res.status} ao buscar /products.json`);
  const { products } = await res.json();

  const catalogo = [];

  for (const p of products) {
    // Kits não são um produto pra vestir no modelo — a foto é da caixa/sacola
    if (/\bkit\b/i.test(p.title)) continue;

    const images = [...p.images].sort((a, b) => a.position - b.position);
    if (!images.length) continue;

    const todasAsFrentes = new Set(p.variants.map((v) => v.featured_image?.src).filter(Boolean));

    const cores = [];
    for (const v of p.variants) {
      const frenteSrc = v.featured_image?.src;
      if (!frenteSrc) continue;

      const idxFrente = images.findIndex((img) => img.src === frenteSrc);
      if (idxFrente === -1) continue;

      const frentesDeOutrasCores = new Set([...todasAsFrentes].filter((s) => s !== frenteSrc));
      cores.push({
        id: String(v.id),
        nome: v.title,
        frente: frenteSrc,
        ladinho: await escolherLadinho(images, idxFrente, frentesDeOutrasCores),
      });
    }

    // Produto sem foto atribuída por variante (ex: cor única) — cai nas duas primeiras
    if (!cores.length) {
      cores.push({
        id: String(p.id),
        nome: p.variants[0]?.title === 'Default Title' ? 'Único' : (p.variants[0]?.title || 'Único'),
        frente: images[0].src,
        ladinho: images[1]?.src || null,
      });
    }

    catalogo.push({
      id: String(p.id),
      titulo: p.title,
      handle: p.handle,
      grupo: grupoDe(p.product_type, p.title),
      subtipo: subtipoDe(p.product_type, p.title),
      cores,
      // todas as fotos do produto, pra poder trocar a escolha na mão no app
      todas: images.map((img) => img.src),
    });
  }

  const dados = {
    loja: STORE,
    sincronizadoEm: new Date().toISOString(),
    produtos: catalogo,
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
  console.log(`catalog.json escrito: ${dados.produtos.length} produtos (${JSON.stringify(porGrupo)}), ${cores} cores`);
}
