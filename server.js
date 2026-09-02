import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import OpenAI, { toFile } from 'openai';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { sincronizarCatalogo } from './sync-catalog.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const upload = multer({ dest: path.join(__dirname, 'uploads') });

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(express.static(path.join(__dirname, 'public')));

async function fileToOpenAI(filePath, mimetype, name) {
  const buffer = fs.readFileSync(filePath);
  return await toFile(buffer, name, { type: mimetype });
}

// Baixa uma foto do CDN da Shopify e entrega no formato que a OpenAI aceita.
// Só aceita URLs do CDN da própria loja — evita a rota virar proxy de download.
async function urlToOpenAI(url, name) {
  if (!/^https:\/\/cdn\.shopify\.com\//.test(url)) {
    throw new Error(`URL de foto não permitida: ${url}`);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Falha ao baixar foto do catálogo (${res.status})`);
  const buffer = Buffer.from(await res.arrayBuffer());
  const mimetype = res.headers.get('content-type') || 'image/jpeg';
  return await toFile(buffer, name, { type: mimetype });
}

// Corrige o fundo para #F3F4F6 exato:
// - Pixels próximos ao fundo (dentro do threshold) → exatamente #F3F4F6
// - Pixels de sombra/óculos (mais escuros) → offset proporcional preservando contraste
async function correctBackground(imageBuffer) {
  const { width, height } = await sharp(imageBuffer).metadata();
  const sz  = 40;
  const TARGET = [243, 244, 246]; // #F3F4F6
  const THRESHOLD = 12; // pixels dentro de 12 unidades do fundo = background puro

  const raw = await sharp(imageBuffer).removeAlpha().raw().toBuffer();

  // Amostra os 4 cantos para obter a cor de referência do fundo
  let sum = [0, 0, 0], count = 0;
  for (const [cx, cy] of [[0,0],[width-sz,0],[0,height-sz],[width-sz,height-sz]]) {
    for (let dy = 0; dy < sz; dy++) {
      for (let dx = 0; dx < sz; dx++) {
        const i = ((cy + dy) * width + (cx + dx)) * 3;
        sum[0] += raw[i]; sum[1] += raw[i+1]; sum[2] += raw[i+2];
        count++;
      }
    }
  }
  const bgRef = sum.map(s => s / count);
  const off   = TARGET.map((t, i) => t - bgRef[i]);

  console.log(`[bg-correct] bgRef=(${bgRef.map(v=>Math.round(v)).join(',')}) off=(${off.map(v=>Math.round(v)).join(',')})`);

  // Flood fill a partir das bordas — só pixels conectados ao fundo real são substituídos
  const isBg = new Uint8Array(width * height);
  const queue = [];

  const seed = (x, y) => {
    const idx = y * width + x;
    if (isBg[idx]) return;
    const i = idx * 3;
    const dist = Math.max(Math.abs(raw[i]-bgRef[0]), Math.abs(raw[i+1]-bgRef[1]), Math.abs(raw[i+2]-bgRef[2]));
    if (dist <= THRESHOLD) { isBg[idx] = 1; queue.push(idx); }
  };

  // Semeia todas as bordas da imagem
  for (let x = 0; x < width;  x++) { seed(x, 0); seed(x, height - 1); }
  for (let y = 0; y < height; y++) { seed(0, y); seed(width - 1, y); }

  // BFS
  let head = 0;
  while (head < queue.length) {
    const idx = queue[head++];
    const x = idx % width, y = Math.floor(idx / width);
    for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const nidx = ny * width + nx;
      if (isBg[nidx]) continue;
      const ni = nidx * 3;
      const dist = Math.max(Math.abs(raw[ni]-bgRef[0]), Math.abs(raw[ni+1]-bgRef[1]), Math.abs(raw[ni+2]-bgRef[2]));
      if (dist <= THRESHOLD) { isBg[nidx] = 1; queue.push(nidx); }
    }
  }

  // Erosão da máscara: remove pixels de fundo vizinhos a pixels de não-fundo
  // Evita comer as bordas dos óculos (2 passes = 2px de margem)
  const erode = (mask) => {
    const out = new Uint8Array(mask);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (!mask[idx]) continue;
        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]]) {
          const nx = x+dx, ny = y+dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          if (!mask[ny * width + nx]) { out[idx] = 0; break; }
        }
      }
    }
    return out;
  };

  const erodedBg = erode(erode(erode(isBg))); // 3 passes = 3px de margem segura

  // Aplica: fundo erodido → #F3F4F6 exato | resto → offset
  for (let idx = 0; idx < width * height; idx++) {
    const i = idx * 3;
    if (erodedBg[idx]) {
      raw[i]   = TARGET[0];
      raw[i+1] = TARGET[1];
      raw[i+2] = TARGET[2];
    } else {
      raw[i]   = Math.min(255, Math.max(0, Math.round(raw[i]   + off[0])));
      raw[i+1] = Math.min(255, Math.max(0, Math.round(raw[i+1] + off[1])));
      raw[i+2] = Math.min(255, Math.max(0, Math.round(raw[i+2] + off[2])));
    }
  }

  return sharp(raw, { raw: { width, height, channels: 3 } }).png().toBuffer();
}

// Versão mais conservadora: só trava em #F3F4F6 os pixels confirmados como fundo puro
// (margem de erosão maior). NÃO aplica offset no resto da imagem — evita distorcer
// pele/cabelo/roupa em fotos com modelo, que têm muito mais detalhe que fotos de produto.
async function lockBackgroundOnly(imageBuffer) {
  const { width, height } = await sharp(imageBuffer).metadata();
  const TARGET = [243, 244, 246];
  const THRESHOLD = 14;

  const raw = await sharp(imageBuffer).removeAlpha().raw().toBuffer();

  const sz = 40;
  let sum = [0, 0, 0], count = 0;
  for (const [cx, cy] of [[0,0],[width-sz,0],[0,height-sz],[width-sz,height-sz]]) {
    for (let dy = 0; dy < sz; dy++) {
      for (let dx = 0; dx < sz; dx++) {
        const i = ((cy + dy) * width + (cx + dx)) * 3;
        sum[0] += raw[i]; sum[1] += raw[i+1]; sum[2] += raw[i+2];
        count++;
      }
    }
  }
  const bgRef = sum.map(s => s / count);

  const isBg = new Uint8Array(width * height);
  const queue = [];
  const seed = (x, y) => {
    const idx = y * width + x;
    if (isBg[idx]) return;
    const i = idx * 3;
    const dist = Math.max(Math.abs(raw[i]-bgRef[0]), Math.abs(raw[i+1]-bgRef[1]), Math.abs(raw[i+2]-bgRef[2]));
    if (dist <= THRESHOLD) { isBg[idx] = 1; queue.push(idx); }
  };
  for (let x = 0; x < width;  x++) { seed(x, 0); seed(x, height - 1); }
  for (let y = 0; y < height; y++) { seed(0, y); seed(width - 1, y); }

  let head = 0;
  while (head < queue.length) {
    const idx = queue[head++];
    const x = idx % width, y = Math.floor(idx / width);
    for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const nidx = ny * width + nx;
      if (isBg[nidx]) continue;
      const ni = nidx * 3;
      const dist = Math.max(Math.abs(raw[ni]-bgRef[0]), Math.abs(raw[ni+1]-bgRef[1]), Math.abs(raw[ni+2]-bgRef[2]));
      if (dist <= THRESHOLD) { isBg[nidx] = 1; queue.push(nidx); }
    }
  }

  // Erosão forte (8 passes = 8px de margem) — protege bordas de cabelo/roupa contra vazamento
  const erode = (mask) => {
    const out = new Uint8Array(mask);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (!mask[idx]) continue;
        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]]) {
          const nx = x+dx, ny = y+dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          if (!mask[ny * width + nx]) { out[idx] = 0; break; }
        }
      }
    }
    return out;
  };
  let eroded = isBg;
  for (let i = 0; i < 8; i++) eroded = erode(eroded);

  for (let idx = 0; idx < width * height; idx++) {
    if (eroded[idx]) {
      const i = idx * 3;
      raw[i] = TARGET[0]; raw[i+1] = TARGET[1]; raw[i+2] = TARGET[2];
    }
  }

  return sharp(raw, { raw: { width, height, channels: 3 } }).png().toBuffer();
}

const PROMPT_FRENTE = `Image 1 is a style reference — follow its background, lighting, shadow, and FRONT-FACING composition exactly.
Images 2+ show the glasses model to use.

Generate a product photo of the glasses from Images 2+ in a FRONT VIEW (straight at the camera), centered, with soft studio lighting and shadow style as Image 1. Reproduce the glasses shape, color, and details precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_LADO = `Image 1 is a style reference — follow its background, lighting, shadow, and SIDE/PROFILE composition exactly.
Images 2+ show the glasses model to use.

Generate a product photo of the glasses from Images 2+ in a SIDE/PROFILE VIEW (like Image 1), with soft studio lighting and shadow style as Image 1. Reproduce the glasses shape, color, and details precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_INCLINADO = `Image 1 is a style reference — follow its background, lighting, shadow, and ANGLED/3-QUARTER composition exactly.
Images 2+ show the glasses model to use.

Generate a product photo of the glasses from Images 2+ in an ANGLED/3-QUARTER VIEW (like Image 1), with soft studio lighting and shadow style as Image 1. Reproduce the glasses shape, color, and details precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const BONE_BRIM_WARNING = `CRITICAL: do NOT shorten, shrink, or alter the length/curvature of the cap's brim (bico/aba) in any way — reproduce the brim at its exact original length and shape as shown in Images 2+. The brim must look normal-length and proportional to the cap, never cropped or truncated short.`;

const PROMPT_BONE_FRENTE = `Image 1 is a style reference — follow its background, lighting, shadow, and FRONT-FACING composition exactly.
Images 2+ show the cap to use.

Generate a product photo of the cap from Images 2+ in a FRONT VIEW (straight at the camera), centered, with soft studio lighting and shadow style as Image 1. Reproduce the cap's shape, fabric texture, color, stitching, logo, and hardware precisely.
${BONE_BRIM_WARNING}
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_BONE_LADINHO = `Image 1 is a style reference — follow its background, lighting, shadow, and ANGLED/3-QUARTER composition exactly.
Images 2+ show the cap to use.

Generate a product photo of the cap from Images 2+ in an ANGLED/3-QUARTER VIEW (like Image 1), with soft studio lighting and shadow style as Image 1. Reproduce the cap's shape, fabric texture, color, stitching, logo, and hardware precisely.
${BONE_BRIM_WARNING}
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_BONE_LADO = `Image 1 is a style reference — follow its background, lighting, shadow, and SIDE/PROFILE composition exactly.
Images 2+ show the cap to use.

Generate a product photo of the cap from Images 2+ in a SIDE/PROFILE VIEW (like Image 1), with soft studio lighting and shadow style as Image 1. Reproduce the cap's shape, fabric texture, color, stitching, logo, strap/buckle, and hardware precisely.
${BONE_BRIM_WARNING} The side view especially must show the brim extending forward at its full, natural length — this is the view where brim length is most visible, so it is critical to get right.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_BONE_TRASEIRA = `Image 1 is a style reference — follow its background, lighting, shadow, and BACK-VIEW composition exactly.
Images 2+ show the cap to use.

Generate a product photo of the cap from Images 2+ in a BACK VIEW (showing the rear strap/closure, like Image 1), centered, with soft studio lighting and shadow style as Image 1. Reproduce the cap's shape, fabric texture, color, stitching, and the rear strap/buckle/snap hardware precisely.
${BONE_BRIM_WARNING}
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

// Configuração dos produtos disponíveis na aba "Óculos/Boné" — cada view tem seu prompt e imagem de referência
const PROMPT_CORRENTE_PENDURADA = `Image 1 is a style reference — follow its background, lighting, shadow, and HANGING/DRAPED composition exactly (chain hanging naturally with the pendant at the bottom).
Images 2+ show the chain/necklace to use.

Generate a product photo of the chain from Images 2+ hanging naturally like Image 1, with soft studio lighting and shadow style as Image 1. Reproduce the chain's link pattern, metal finish, color, clasp, and pendant (if any) precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_CORRENTE_COMPLETA = `Image 1 is a style reference — follow its background, lighting, shadow, and FULL FLAT-LAY COMPOSITION exactly (chain laid out in a full circle/loop, viewed from directly above).
Images 2+ show the chain/necklace to use.

Generate a product photo of the chain from Images 2+ laid out flat in a full circle like Image 1, with soft studio lighting and shadow style as Image 1. Reproduce the chain's link pattern, metal finish, color, clasp, and pendant (if any) precisely, including the full length of the chain.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_CORRENTE_TEXTURA = `Image 1 is a style reference — follow its background, lighting, shadow, and CLOSE-UP MACRO composition exactly (tight close-up on a diagonal segment of the chain, showing link texture in detail).
Images 2+ show the chain/necklace to use.

Generate a close-up macro product photo of the chain from Images 2+ like Image 1, with soft studio lighting and shadow style as Image 1. Reproduce the chain's exact link pattern, metal finish, color, and texture in fine detail.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_BRACELETE_COMPLETA = `Image 1 is a style reference — follow its background, lighting, shadow, and FULL FLAT-LAY COMPOSITION exactly (bracelet laid out in a full circle/loop, viewed from directly above).
Images 2+ show the bracelet to use.

Generate a product photo of the bracelet from Images 2+ laid out flat in a full circle like Image 1, with soft studio lighting and shadow style as Image 1. Reproduce the bracelet's link pattern, metal finish, color, and clasp precisely, including the full length of the bracelet.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_BRACELETE_TEXTURA = `Image 1 is a style reference — follow its background, lighting, shadow, and CLOSE-UP MACRO composition exactly (tight close-up on the clasp/lobster-clasp area of the bracelet, showing link texture in detail).
Images 2+ show the bracelet to use.

Generate a close-up macro product photo of the bracelet from Images 2+ like Image 1, with soft studio lighting and shadow style as Image 1. Reproduce the bracelet's exact link pattern, metal finish, color, clasp, and texture in fine detail.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_ANEL_ANGULO1 = `Image 1 is a style reference — follow its background, lighting, shadow, camera angle, and framing exactly (ring standing upright, viewed from a 3/4 angle showing the front and side of the band).
Images 2+ show the ring to use.

Generate a product photo of the ring from Images 2+ positioned and angled exactly like Image 1, with soft studio lighting and shadow style as Image 1. Reproduce the ring's exact shape, metal finish, color, and any engraving or texture precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_ANEL_ANGULO2 = `Image 1 is a style reference — follow its background, lighting, shadow, camera angle, and framing exactly (ring standing upright, viewed from a lower front angle showing the inner engraving detail).
Images 2+ show the ring to use.

Generate a product photo of the ring from Images 2+ positioned and angled exactly like Image 1, with soft studio lighting and shadow style as Image 1. Reproduce the ring's exact shape, metal finish, color, and any engraving or texture precisely.
IMPORTANT: Image 1 has text engraved on the inner band of ITS ring (e.g. "STAINLESS STEEL") — this text belongs only to the reference ring and must NOT be copied. Do not add any text or inscription to the inner band unless it is actually present on the ring from Images 2+.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PRODUCTS = {
  oculos: {
    defaultView: 'frente',
    views: {
      frente:    { prompt: PROMPT_FRENTE,    refPath: 'public/references/frente.png',    label: 'VISTA FRONTAL' },
      inclinado: { prompt: PROMPT_INCLINADO, refPath: 'public/references/inclinado.png', label: 'VISTA INCLINADA' },
      lado:      { prompt: PROMPT_LADO,      refPath: 'public/references/lado.png',      label: 'VISTA LATERAL' },
    },
  },
  bone: {
    defaultView: 'frente',
    views: {
      frente:   { prompt: PROMPT_BONE_FRENTE,   refPath: 'public/references/bone-frente.png',   label: 'VISTA FRONTAL' },
      ladinho:  { prompt: PROMPT_BONE_LADINHO,  refPath: 'public/references/bone-ladinho.png',  label: 'VISTA 3/4' },
      lado:     { prompt: PROMPT_BONE_LADO,     refPath: 'public/references/bone-lado.png',     label: 'VISTA LATERAL' },
      traseira: { prompt: PROMPT_BONE_TRASEIRA, refPath: 'public/references/bone-traseira.png', label: 'VISTA TRASEIRA' },
    },
  },
  corrente: {
    defaultView: 'pendurada',
    views: {
      pendurada: { prompt: PROMPT_CORRENTE_PENDURADA, refPath: 'public/references/corrente-pendurada.png', label: 'PENDURADA' },
      completa:  { prompt: PROMPT_CORRENTE_COMPLETA,  refPath: 'public/references/corrente-completa.png',  label: 'VISTA COMPLETA' },
      textura:   { prompt: PROMPT_CORRENTE_TEXTURA,   refPath: 'public/references/corrente-textura.png',   label: 'TEXTURA (CLOSE-UP)' },
    },
  },
  bracelete: {
    defaultView: 'completa',
    views: {
      completa: { prompt: PROMPT_BRACELETE_COMPLETA, refPath: 'public/references/bracelete-completa.png', label: 'VISTA COMPLETA' },
      textura:  { prompt: PROMPT_BRACELETE_TEXTURA,  refPath: 'public/references/bracelete-textura.png',  label: 'TEXTURA (CLOSE-UP)' },
    },
  },
  anel: {
    defaultView: 'angulo1',
    views: {
      angulo1: { prompt: PROMPT_ANEL_ANGULO1, refPath: 'public/references/anel-1.png', label: 'ÂNGULO 1' },
      angulo2: { prompt: PROMPT_ANEL_ANGULO2, refPath: 'public/references/anel-2.png', label: 'ÂNGULO 2' },
    },
  },
};

// Monta prompt dinamicamente conforme os extras selecionados
function buildModelPrompt(glassesCount, outfitIdx, expressionIdx, boneStartIdx, boneCount) {
  // Monta descrição dos índices de óculos (2 a 1+glassesCount)
  const glassesStart = 2;
  const glassesEnd   = 1 + glassesCount;
  const glassesRef   = glassesCount === 1
    ? `Image ${glassesStart} shows the glasses`
    : `Images ${glassesStart} to ${glassesEnd} show the glasses from different angles — use all of them as reference to understand the exact shape, color, lenses, and frame`;

  const boneEnd = boneStartIdx ? boneStartIdx + boneCount - 1 : null;
  const boneRef = boneStartIdx
    ? (boneCount === 1 ? `Image ${boneStartIdx} shows the cap` : `Images ${boneStartIdx} to ${boneEnd} show the cap from different angles — use all of them`)
    : null;

  const header = [`Image 1 is the model reference photo. ${glassesRef}. Image ${outfitIdx} shows the outfit.`];
  if (expressionIdx) header[0] += ` Image ${expressionIdx} is a facial expression reference.`;
  if (boneRef)        header[0] += ` ${boneRef} to understand its full shape.`;

  const lines = [...header, ''];

  lines.push(`Generate a professional fashion photo where the model from Image 1 is wearing the glasses from ${glassesCount === 1 ? `Image ${glassesStart}` : `Images ${glassesStart}-${glassesEnd}`} and the clothing from Image ${outfitIdx}.`);
  lines.push('- Preserve the model\'s face, skin, and hair exactly as in Image 1');
  lines.push('- Allow only very subtle natural variation: slight micro-expression shift and minor hair strand movement — to create a natural feel');

  if (expressionIdx) {
    lines.push(`- Replicate only the facial expression from Image ${expressionIdx} (mouth position, eye openness, brow shape) onto ${varias ? 'every person' : 'the model'} — do NOT copy the face, identity, skin tone or any other feature of the person in Image ${expressionIdx}`);
  }

  lines.push(`- Place the glasses naturally and precisely on the model\'s face, preserving their exact shape, color, lenses, and frame${glassesCount > 1 ? ' — cross-reference all glasses images to get the details right' : ''}`);
  lines.push(`- Dress the model in the exact outfit shown in Image ${outfitIdx}`);

  if (boneStartIdx) {
    const boneImgRef = boneCount === 1 ? `Image ${boneStartIdx}` : `Images ${boneStartIdx}-${boneEnd}`;
    lines.push(`- Place the cap shown in ${boneImgRef} naturally on the model\'s head, fitting the pose and angle — preserve its exact shape, color, and details`);
  }

  lines.push('- Professional studio lighting, soft and clean');
  lines.push('IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture.');

  return lines.join('\n');
}

// Monta prompt para a seção "Criativos": recria a foto de referência trocando o modelo
// Quantas fotos de produto cabem numa geração — dá pra combinar óculos + boné +
// joia de uma vez, cada um entrando com frente e ladinho
const MAX_FOTOS_PRODUTO = 10;

// Quantas pessoas cabem numa mesma cena
const MAX_MODELOS = 4;

const NOME_DO_GRUPO = {
  oculos: 'the sunglasses',
  bone:   'the cap',
  joia:   'the jewelry piece',
};

// Cada tipo de joia é usado num lugar do corpo — sem isso a IA põe pulseira no pescoço
const NOME_DO_SUBTIPO = {
  anel:     { nome: 'the ring',     onde: "on the model's finger" },
  pulseira: { nome: 'the bracelet', onde: "around the model's wrist" },
  corrente: { nome: 'the chain',    onde: "around the model's neck" },
};

function nomeDoProduto(item) {
  return NOME_DO_SUBTIPO[item.subtipo]?.nome || NOME_DO_GRUPO[item.grupo] || 'the product';
}

function descricaoDoProduto(item) {
  const cor = item.cor && item.cor !== 'Único' ? ` in the "${item.cor}" colorway` : '';
  return `${nomeDoProduto(item)} "${item.titulo}"${cor}`;
}

function comoVestirProduto(item) {
  const nome = nomeDoProduto(item);
  if (item.grupo === 'oculos') {
    return `Place ${nome} naturally and precisely on the model's face, replacing whatever glasses (if any) appear in Image 1`;
  }
  if (item.grupo === 'bone') {
    return `Put ${nome} on the model's head at a natural angle, replacing whatever headwear (if any) appears in Image 1`;
  }
  if (item.grupo === 'joia') {
    const onde = NOME_DO_SUBTIPO[item.subtipo]?.onde;
    return onde
      ? `Have the model wear ${nome} ${onde} — it must appear there and nowhere else, replacing any equivalent jewelry in Image 1. If the framing of Image 1 does not show that part of the body, widen nothing and simply leave it out rather than moving it elsewhere`
      : `Have the model wear ${nome} in its natural position on the body, replacing any equivalent jewelry in Image 1`;
  }
  return `Place ${nome} naturally on the model`;
}

// produtos: [{ descricao, comoVestir, inicio, fim }] — um por produto anexado,
// com a faixa de imagens que pertence a ele
// modelos: [{ idx }] — uma imagem de referência por pessoa que deve aparecer na cena
function buildCreativePrompt(modelos, produtos, keepOutfit, outfitIdx, expressionIdx, extraText) {
  const varias   = modelos.length > 1;
  const idxs     = modelos.map(m => m.idx);
  const listaIdx = varias
    ? `Images ${idxs.slice(0, -1).join(', ')} and ${idxs[idxs.length - 1]}`
    : `Image ${idxs[0]}`;

  const produtoRef = produtos.map(p => (
    p.inicio === p.fim
      ? `Image ${p.inicio} shows ${p.descricao}`
      : `Images ${p.inicio} to ${p.fim} show ${p.descricao} from different angles — use all of them as reference to understand its exact shape, color and details`
  )).join('. ');

  const refModelos = varias
    ? `${listaIdx} are the model reference photos — one per person, ${modelos.length} people in total`
    : `${listaIdx} is the model reference photo`;
  const header = [`Image 1 is the reference photo — replicate its pose, camera angle, framing, background, lighting, and shadow style exactly. ${refModelos}. ${produtoRef}.`];
  if (!keepOutfit && outfitIdx) header[0] += ` Image ${outfitIdx} shows the outfit to use.`;
  if (expressionIdx) header[0] += ` Image ${expressionIdx} is a facial expression reference.`;

  const lines = [...header, ''];

  if (varias) {
    lines.push(`Generate a photo that recreates Image 1 exactly, but with ${modelos.length} people in it — one for each model reference (${listaIdx}).`);
    lines.push(`- Every one of the ${modelos.length} models must appear in the final photo, each as a distinct person — never merge two references into one face and never repeat the same face twice`);
    lines.push('- If Image 1 shows fewer people than that, add the missing ones into the scene naturally, side by side, matching its framing, scale, lighting and shadows');
    modelos.forEach((m, i) => lines.push(`- Person ${i + 1}: preserve the face, skin, and hair exactly as in Image ${m.idx}`));
  } else {
    lines.push(`Generate a photo that recreates Image 1 exactly, but replace the person in it with the model from Image ${idxs[0]}.`);
    lines.push(`- Preserve the model's face, skin, and hair exactly as in Image ${idxs[0]}`);
  }
  lines.push('- Allow only very subtle natural variation: slight micro-expression shift and minor hair strand movement — to create a natural feel');

  if (expressionIdx) {
    lines.push(`- Replicate only the facial expression from Image ${expressionIdx} (mouth position, eye openness, brow shape) onto ${varias ? 'every person' : 'the model'} — do NOT copy the face, identity, skin tone or any other feature of the person in Image ${expressionIdx}`);
  }

  for (const p of produtos) {
    const varias = p.fim > p.inicio ? ' — cross-reference all of its images to get the details right' : '';
    lines.push(`- ${p.comoVestir}, preserving its exact shape, color and details${varias}`);
  }
  if (produtos.length > 1) {
    lines.push(`- All the products above must appear together at the same time, worn naturally and consistently${varias ? ' — spread them across the people so each product is clearly visible on someone' : ' on the model'}`);
  }

  if (keepOutfit) {
    lines.push(`- Keep the exact same clothing/outfit style worn by the person in Image 1, now on the new ${varias ? 'models' : 'model'}`);
  } else {
    lines.push(`- Dress ${varias ? 'every model' : 'the model'} in the exact outfit shown in Image ${outfitIdx}`);
  }

  lines.push('- Keep everything else from Image 1 identical: pose, camera angle, framing, background, lighting, and shadows');

  if (extraText && extraText.trim()) {
    lines.push(`- Additional instructions: ${extraText.trim()}`);
  }

  return lines.join('\n');
}

// Monta prompt para "Corrente com Modelo": recria a cena de referência trocando a pessoa
// e substituindo TODAS as correntes da foto original por uma única (a anexada)
function buildCorrenteModeloPrompt(chainCount, keepOutfit, outfitIdx, extraText, jacketIdx, jacketMode) {
  const modelIdx   = 2;
  const chainStart = 3;
  const chainEnd   = 2 + chainCount;
  const chainRef   = chainCount === 1
    ? `Image ${chainStart} shows the chain`
    : `Images ${chainStart} to ${chainEnd} show the chain from different angles — use all of them as reference to understand its exact shape, color, and finish`;

  const header = [`Image 1 is the reference photo — replicate its pose, camera angle, framing, background, lighting, and shadow style exactly. Image ${modelIdx} is the model reference photo (skin tone, build). ${chainRef}.`];
  if (!keepOutfit && outfitIdx) header[0] += ` Image ${outfitIdx} shows the outfit to use.`;
  if (jacketIdx) header[0] += ` Image ${jacketIdx} shows a jacket/coat to add to the outfit.`;

  const lines = [...header, ''];

  lines.push(`Generate a photo that recreates Image 1 exactly, but replace the person's skin/neck/body with the model from Image ${modelIdx}, matching their skin tone and build naturally.`);

  if (keepOutfit) {
    lines.push(`- Keep the exact same clothing/outfit style worn by the person in Image 1, now on the new ${varias ? 'models' : 'model'}`);
  } else {
    lines.push(`- Dress ${varias ? 'every model' : 'the model'} in the exact outfit shown in Image ${outfitIdx}`);
  }

  if (jacketIdx) {
    if (jacketMode === 'replace') {
      lines.push(`- Replace the outer clothing/garment the person is currently wearing with the jacket/coat shown in Image ${jacketIdx} — swap it in completely, matching its exact color, fabric, texture, and design`);
    } else {
      lines.push(`- Add the jacket/coat shown in Image ${jacketIdx} on top of the existing outfit, layered over it — matching its exact color, fabric, texture, and design`);
    }
    lines.push('- IMPORTANT: the jacket/coat must always be worn OPEN/unzipped/unbuttoned, never closed, so the clothing/chain underneath stays fully visible');
    lines.push('- Fit the jacket/coat naturally to the model\'s pose and body, with realistic drape, folds, and shadows');
  }

  lines.push(`- IMPORTANT: if Image 1 shows more than one chain/necklace layered together, replace ALL of them with a single chain — do not keep any of the original chains from Image 1, only the new one from ${chainCount === 1 ? `Image ${chainStart}` : `Images ${chainStart}-${chainEnd}`}`);
  lines.push('- Place the new chain naturally around the neck, following the same drape, length, and position style as the original chain(s) in Image 1');
  lines.push('- Preserve the chain\'s exact link pattern, metal finish, color, clasp, and pendant (if any) precisely');
  lines.push('- Keep everything else from Image 1 identical: pose, camera angle, framing, background, lighting, and shadows');

  if (extraText && extraText.trim()) {
    lines.push(`- Additional instructions: ${extraText.trim()}`);
  }

  return lines.join('\n');
}

// Para cada referência: como o casaco/jaqueta deve entrar quando anexado.
// 'replace' = troca a peça externa atual pelo casaco anexado (ex: foto 2)
// 'add'     = coloca o casaco por cima da roupa já existente (ex: fotos 1 e 3)
const CORRENTE_MODELO_JACKET_MODE = {
  colar1: 'add',
  colar2: 'replace',
  colar3: 'add',
  colar4: 'add',
};

const CORRENTE_MODELO_REFS = {
  colar1: 'public/references/corrente-modelo-1.jpg',
  colar2: 'public/references/corrente-modelo-2.jpg',
  colar3: 'public/references/corrente-modelo-3.jpg',
  colar4: 'public/references/corrente-modelo-4.jpg',
};

// Monta prompt para "Bracelete com Modelo": recria a cena de referência trocando a pessoa
// (para variar o braço/mão) e substituindo o bracelete original pelo anexado.
// Permite ainda trocar camisa e/ou calça — só entra instrução se algo for anexado.
function buildBraceleteModeloPrompt(braceletCount, shirtIdx, pantsIdx, extraText) {
  const modelIdx      = 2;
  const braceletStart = 3;
  const braceletEnd   = 2 + braceletCount;
  const braceletRef   = braceletCount === 1
    ? `Image ${braceletStart} shows the bracelet`
    : `Images ${braceletStart} to ${braceletEnd} show the bracelet from different angles — use all of them as reference to understand its exact shape, color, and finish`;

  const header = [`Image 1 is the reference photo — replicate its pose, camera angle, framing, background, and lighting exactly. Image ${modelIdx} is the model reference photo (skin tone, hand/arm build). ${braceletRef}.`];
  if (shirtIdx) header[0] += ` Image ${shirtIdx} shows the shirt/clothing to use.`;
  if (pantsIdx) header[0] += ` Image ${pantsIdx} shows the pants to use.`;

  const lines = [...header, ''];

  lines.push(`Generate a photo that recreates Image 1 exactly, but replace the person's hand/arm/skin with the model from Image ${modelIdx}, matching their skin tone and build naturally.`);

  if (shirtIdx) {
    lines.push(`- Dress the model in the exact shirt/clothing shown in Image ${shirtIdx}`);
  }

  if (pantsIdx) {
    lines.push(`- Dress the model in the exact pants shown in Image ${pantsIdx}`);
  }

  lines.push(`- IMPORTANT: replace the bracelet the person is wearing in Image 1 entirely with the one from ${braceletCount === 1 ? `Image ${braceletStart}` : `Images ${braceletStart}-${braceletEnd}`} — do not keep any part of the original bracelet`);
  lines.push('- Place the new bracelet naturally around the wrist, following the same position and fit style as the original bracelet in Image 1');
  lines.push('- Preserve the bracelet\'s exact link pattern, metal finish, color, and clasp precisely');
  lines.push('- If Image 1 shows other jewelry (rings, chains) not being replaced, keep them exactly as they are');
  lines.push('- Keep everything else from Image 1 identical: pose, camera angle, framing, and shadow style');

  if (extraText && extraText.trim()) {
    lines.push(`- Additional instructions: ${extraText.trim()}`);
  }

  return lines.join('\n');
}

// visible: quais elementos (camisa/calça) aparecem em cada foto de referência,
// usado pelo front pra só mostrar a opção de trocar quando fizer sentido.
const BRACELETE_MODELO_VISIBLE = {
  pulso1: { shirt: true,  pants: false },
  pulso2: { shirt: false, pants: false },
  pulso3: { shirt: true,  pants: true  },
  pulso4: { shirt: true,  pants: false },
};

const BRACELETE_MODELO_REFS = {
  pulso1: 'public/references/bracelete-modelo-1.png',
  pulso2: 'public/references/bracelete-modelo-2.png',
  pulso3: 'public/references/bracelete-modelo-3.png',
  pulso4: 'public/references/bracelete-modelo-4.png',
};

// Monta prompt para "Anel com Modelo": recria a cena de referência trocando a pessoa
// (para variar a mão) e substituindo o anel original pelo anexado.
// Permite ainda trocar camisa e/ou calça — só entra instrução se algo for anexado.
function buildAnelModeloPrompt(ringCount, shirtIdx, pantsIdx, extraText) {
  const modelIdx  = 2;
  const ringStart = 3;
  const ringEnd   = 2 + ringCount;
  const ringRef   = ringCount === 1
    ? `Image ${ringStart} shows the ring`
    : `Images ${ringStart} to ${ringEnd} show the ring from different angles — use all of them as reference to understand its exact shape, color, and finish`;

  const header = [`Image 1 is the reference photo — replicate its pose, camera angle, framing, background, and lighting exactly. Image ${modelIdx} is the model reference photo (skin tone, hand build). ${ringRef}.`];
  if (shirtIdx) header[0] += ` Image ${shirtIdx} shows the shirt/clothing to use.`;
  if (pantsIdx) header[0] += ` Image ${pantsIdx} shows the pants to use.`;

  const lines = [...header, ''];

  lines.push(`Generate a photo that recreates Image 1 exactly, but replace the person's hand/skin with the model from Image ${modelIdx}, matching their skin tone and build naturally.`);

  if (shirtIdx) {
    lines.push(`- Dress the model in the exact shirt/clothing shown in Image ${shirtIdx}`);
  }

  if (pantsIdx) {
    lines.push(`- Dress the model in the exact pants shown in Image ${pantsIdx}`);
  }

  lines.push(`- IMPORTANT: replace the ring the person is wearing in Image 1 entirely with the one from ${ringCount === 1 ? `Image ${ringStart}` : `Images ${ringStart}-${ringEnd}`} — do not keep any part of the original ring`);
  lines.push('- Place the new ring naturally on the same finger, following the same position and fit style as the original ring in Image 1');
  lines.push('- Preserve the ring\'s exact shape, metal finish, color, and any engraving or texture precisely');
  lines.push('- If Image 1 shows other jewelry (bracelets, chains) not being replaced, keep them exactly as they are');
  lines.push('- Keep everything else from Image 1 identical: pose, camera angle, framing, and shadow style');

  if (extraText && extraText.trim()) {
    lines.push(`- Additional instructions: ${extraText.trim()}`);
  }

  return lines.join('\n');
}

const ANEL_MODELO_VISIBLE = {
  anel1: { shirt: true,  pants: true  },
  anel2: { shirt: false, pants: false },
  anel3: { shirt: true,  pants: false },
  anel4: { shirt: true,  pants: false },
};

const ANEL_MODELO_REFS = {
  anel1: 'public/references/anel-modelo-1.png',
  anel2: 'public/references/anel-modelo-2.png',
  anel3: 'public/references/anel-modelo-3.png',
  anel4: 'public/references/anel-modelo-4.png',
};

const PROMPT_SOMBRA = `Generate a clean professional product photo of the glasses shown in the images.
- Glasses: front view, horizontally centered
- Soft drop shadow directly beneath the glasses
- Professional studio lighting
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

app.post('/api/generate', upload.array('images', 10), async (req, res) => {
  const uploadedPaths = (req.files || []).map(f => f.path);
  try {
    const { view, produto } = req.body;
    if (!req.files || req.files.length === 0)
      return res.status(400).json({ error: 'Nenhuma imagem enviada.' });

    const product = PRODUCTS[produto] || PRODUCTS.oculos;
    const viewConfig = product.views[view] || product.views[product.defaultView];
    if (!viewConfig) return res.status(400).json({ error: 'Vista inválida para este produto.' });

    const prompt  = viewConfig.prompt;
    const refPath = path.join(__dirname, viewConfig.refPath);

    const refFile = await fileToOpenAI(refPath, 'image/png', `ref-${view}.png`);
    const productFiles = await Promise.all(
      req.files.map((f, i) => fileToOpenAI(f.path, f.mimetype, `produto-${i + 1}.${f.mimetype.split('/')[1] || 'jpg'}`))
    );

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    console.log(`[generate] produto=${produto || 'oculos'} view=${view} ref + ${productFiles.length} produto(s)`);

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: [refFile, ...productFiles],
      prompt,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    const corrected = await correctBackground(Buffer.from(b64, 'base64'));
    res.json({ image: corrected.toString('base64') });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Expõe as views disponíveis por produto (chave + label), pra montar os botões no front
app.get('/api/products', (req, res) => {
  const result = {};
  for (const [key, product] of Object.entries(PRODUCTS)) {
    result[key] = {
      defaultView: product.defaultView,
      views: Object.entries(product.views).map(([viewKey, v]) => ({
        key: viewKey,
        label: v.label,
        thumb: '/' + v.refPath.replace(/^public\//, ''),
      })),
    };
  }
  res.json(result);
});

// Catálogo da loja Shopify — produtos com a foto de frente e a de ladinho de cada cor
app.get('/api/catalog', (req, res) => {
  const catalogPath = path.join(__dirname, 'public/catalog.json');
  if (!fs.existsSync(catalogPath)) {
    return res.status(404).json({ error: 'Catálogo ainda não sincronizado. Clique em "Sincronizar".' });
  }
  res.json(JSON.parse(fs.readFileSync(catalogPath, 'utf8')));
});

app.post('/api/catalog/sync', async (req, res) => {
  try {
    const dados = await sincronizarCatalogo();
    console.log(`[catalog] sincronizado: ${dados.produtos.length} produtos`);
    res.json(dados);
  } catch (err) {
    console.error('[catalog] erro na sincronização:', err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/png-sombra', upload.array('images', 10), async (req, res) => {
  const uploadedPaths = (req.files || []).map(f => f.path);
  try {
    if (!req.files || req.files.length === 0)
      return res.status(400).json({ error: 'Nenhuma imagem enviada.' });

    const productFiles = await Promise.all(
      req.files.map((f, i) => fileToOpenAI(f.path, f.mimetype, `oculos-${i + 1}.${f.mimetype.split('/')[1] || 'jpg'}`))
    );

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    console.log(`[png-sombra] ${productFiles.length} imagem(ns)`);

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: productFiles,
      prompt: PROMPT_SOMBRA,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    const corrected = await correctBackground(Buffer.from(b64, 'base64'));
    res.json({ image: corrected.toString('base64') });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Lista modelos disponíveis em public/models/
// Convenção: nome-frente.jpg e nome-ladinho.jpg
app.get('/api/models', (req, res) => {
  const modelsDir = path.join(__dirname, 'public/models');
  const exts = ['.jpg', '.jpeg', '.png', '.webp'];
  const files = fs.readdirSync(modelsDir)
    .filter(f => exts.includes(path.extname(f).toLowerCase()));

  const map = {};
  files.forEach(f => {
    const base = path.basename(f, path.extname(f));
    const mFrente  = base.match(/^(.+)-frente$/i);
    const mLadinho = base.match(/^(.+)-ladinho$/i);
    if (mFrente) {
      const name = mFrente[1];
      if (!map[name]) map[name] = { name };
      map[name].frente = f;
    } else if (mLadinho) {
      const name = mLadinho[1];
      if (!map[name]) map[name] = { name };
      map[name].ladinho = f;
    } else {
      if (!map[base]) map[base] = { name: base };
      map[base].frente = f;
    }
  });

  res.json(Object.values(map).sort((a, b) => a.name.localeCompare(b.name)));
});

// Lista expressões disponíveis em public/expressions/
app.get('/api/expressions', (req, res) => {
  const dir = path.join(__dirname, 'public/expressions');
  const exts = ['.jpg', '.jpeg', '.png', '.webp'];
  if (!fs.existsSync(dir)) return res.json([]);
  const files = fs.readdirSync(dir)
    .filter(f => exts.includes(path.extname(f).toLowerCase()))
    .map(f => ({ file: f, name: path.basename(f, path.extname(f)) }));
  res.json(files);
});

const modelUpload = multer({ dest: path.join(__dirname, 'uploads') });

app.post('/api/generate-model', modelUpload.fields([
  { name: 'glasses', maxCount: 5 },
  { name: 'clothing', maxCount: 1 },
  { name: 'customBone', maxCount: 2 },
]), async (req, res) => {
  const glassesFiles = req.files?.['glasses'] || [];
  const clothingFile = req.files?.['clothing']?.[0];
  const customBoneFiles = req.files?.['customBone'] || [];
  const uploadedPaths = [...glassesFiles.map(f => f.path), clothingFile?.path, ...customBoneFiles.map(f => f.path)].filter(Boolean);
  try {
    const { modelFile, pose } = req.body;
    if (!glassesFiles.length) return res.status(400).json({ error: 'Envie ao menos uma foto dos óculos.' });
    if (!clothingFile)        return res.status(400).json({ error: 'Envie a foto da roupa.' });
    if (!modelFile)           return res.status(400).json({ error: 'Selecione um modelo.' });

    const modelPath = path.join(__dirname, 'public/models', modelFile);
    if (!fs.existsSync(modelPath))
      return res.status(400).json({ error: 'Modelo não encontrado.' });

    const { expressionFile, boneSelected } = req.body;
    const ext = path.extname(modelFile).toLowerCase();
    const mime = ext === '.png' ? 'image/png' : 'image/jpeg';

    // Image 1: modelo
    const modelRef = await fileToOpenAI(modelPath, mime, 'model.jpg');
    const images = [modelRef];

    // Images 2 a N+1: óculos (1 ou mais)
    for (let i = 0; i < glassesFiles.length; i++) {
      images.push(await fileToOpenAI(glassesFiles[i].path, glassesFiles[i].mimetype, `glasses-${i+1}.jpg`));
    }
    const glassesEndIdx = images.length; // índice da última imagem de óculos

    // Image N+2: roupa
    const clothingRef = await fileToOpenAI(clothingFile.path, clothingFile.mimetype, 'clothing.jpg');
    images.push(clothingRef);
    const outfitIdx = images.length;

    let expressionIdx = null;
    let boneStartIdx  = null;
    let boneCount     = 0;

    // Expressão (opcional)
    if (expressionFile) {
      const exprPath = path.join(__dirname, 'public/expressions', expressionFile);
      if (fs.existsSync(exprPath)) {
        const exprMime = path.extname(expressionFile).toLowerCase() === '.png' ? 'image/png' : 'image/jpeg';
        images.push(await fileToOpenAI(exprPath, exprMime, 'expression.jpg'));
        expressionIdx = images.length;
      }
    }

    // Boné (opcional) — anexado pelo usuário tem prioridade sobre o boné embutido do site
    if (customBoneFiles.length) {
      for (let i = 0; i < customBoneFiles.length; i++) {
        images.push(await fileToOpenAI(customBoneFiles[i].path, customBoneFiles[i].mimetype, `bone-custom-${i + 1}.jpg`));
      }
      boneStartIdx = images.length - customBoneFiles.length + 1;
      boneCount = customBoneFiles.length;
    } else if (boneSelected === 'true') {
      const boneFrente = path.join(__dirname, 'public/accessories/bone-frente.png');
      const boneLado   = path.join(__dirname, 'public/accessories/bone-lado.png');
      if (fs.existsSync(boneFrente) && fs.existsSync(boneLado)) {
        images.push(await fileToOpenAI(boneFrente, 'image/png', 'bone-frente.png'));
        boneStartIdx = images.length;
        images.push(await fileToOpenAI(boneLado,   'image/png', 'bone-lado.png'));
        boneCount = 2;
      }
    }

    const prompt = buildModelPrompt(glassesFiles.length, outfitIdx, expressionIdx, boneStartIdx, boneCount);
    console.log(`[generate-model] model=${modelFile} pose=${pose} glasses=${glassesFiles.length} expr=${!!expressionFile} bone=${customBoneFiles.length ? 'custom(' + customBoneFiles.length + ')' : boneSelected}`);

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: images,
      prompt,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    // Trava só o fundo confirmado em #F3F4F6 (sem distorcer pele/cabelo/roupa)
    const corrected = await lockBackgroundOnly(Buffer.from(b64, 'base64'));
    res.json({ image: corrected.toString('base64') });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const creativeUpload = multer({ dest: path.join(__dirname, 'uploads') });

app.post('/api/generate-creative', creativeUpload.fields([
  { name: 'reference', maxCount: 1 },
  { name: 'glasses', maxCount: 5 },
  { name: 'outfit', maxCount: 1 },
]), async (req, res) => {
  const referenceFile = req.files?.['reference']?.[0];
  const glassesFiles   = req.files?.['glasses'] || [];
  const outfitFile     = req.files?.['outfit']?.[0];
  const uploadedPaths  = [referenceFile?.path, ...glassesFiles.map(f => f.path), outfitFile?.path].filter(Boolean);
  try {
    const { modelFile, pose, keepOutfit, expressionFile, extraText, size } = req.body;

    // Modelos: dá pra colocar mais de uma pessoa na mesma cena. Aceita tanto
    // modelFiles (lista) quanto o modelFile antigo, de uma pessoa só.
    let modelFiles = [];
    if (req.body.modelFiles) {
      try {
        modelFiles = JSON.parse(req.body.modelFiles);
        if (!Array.isArray(modelFiles) || modelFiles.some(f => typeof f !== 'string')) throw new Error();
      } catch {
        return res.status(400).json({ error: 'modelFiles inválido.' });
      }
    } else if (modelFile) {
      modelFiles = [modelFile];
    }

    // Produtos escolhidos no catálogo da loja — podem ser vários ao mesmo tempo
    // (ex: óculos + boné + corrente), e somam com as fotos anexadas na mão
    let catalogItems = [];
    if (req.body.catalogItems) {
      try {
        catalogItems = JSON.parse(req.body.catalogItems);
        if (!Array.isArray(catalogItems) || catalogItems.some(i => !Array.isArray(i?.urls))) throw new Error();
      } catch {
        return res.status(400).json({ error: 'catalogItems inválido.' });
      }
    }
    const totalCatalogo = catalogItems.reduce((n, i) => n + i.urls.length, 0);
    const totalProduto  = glassesFiles.length + totalCatalogo;

    if (!referenceFile)   return res.status(400).json({ error: 'Envie a foto de referência.' });
    if (!totalProduto)    return res.status(400).json({ error: 'Selecione um produto do catálogo ou anexe ao menos uma foto.' });
    if (totalProduto > MAX_FOTOS_PRODUTO) return res.status(400).json({ error: `No máximo ${MAX_FOTOS_PRODUTO} fotos de produto.` });
    if (!modelFiles.length) return res.status(400).json({ error: 'Selecione um modelo.' });
    if (modelFiles.length > MAX_MODELOS) return res.status(400).json({ error: `No máximo ${MAX_MODELOS} pessoas por criativo.` });
    // nomes vêm do próprio grid de modelos; barra qualquer tentativa de sair da pasta
    if (modelFiles.some(f => /[\\/]/.test(f) || f.includes('..')))
      return res.status(400).json({ error: 'Nome de modelo inválido.' });

    const allowedSizes = ['1024x1024', '1024x1536', '1536x1024'];
    const imageSize = allowedSizes.includes(size) ? size : '1024x1024';

    const keepOutfitBool = keepOutfit === 'true';
    if (!keepOutfitBool && !outfitFile)
      return res.status(400).json({ error: 'Envie a foto da roupa ou selecione "manter roupa da referência".' });

    const modelPaths = modelFiles.map(f => path.join(__dirname, 'public/models', f));
    const faltando = modelFiles.filter((f, i) => !fs.existsSync(modelPaths[i]));
    if (faltando.length)
      return res.status(400).json({ error: `Modelo não encontrado: ${faltando.join(', ')}` });

    // Image 1: referência
    const referenceRef = await fileToOpenAI(referenceFile.path, referenceFile.mimetype, 'reference.jpg');
    const images = [referenceRef];

    // Images 2..N+1: uma foto por pessoa que deve aparecer na cena
    const modelos = [];
    for (let i = 0; i < modelFiles.length; i++) {
      const mime = path.extname(modelFiles[i]).toLowerCase() === '.png' ? 'image/png' : 'image/jpeg';
      images.push(await fileToOpenAI(modelPaths[i], mime, `model-${i + 1}.jpg`));
      modelos.push({ idx: images.length });
    }

    // Images 3 a N+2: produtos — primeiro os do catálogo, depois as fotos da mão.
    // Guarda a faixa de imagens de cada produto pra o prompt saber o que é o quê.
    const produtos = [];
    let contador = 0;
    for (const item of catalogItems) {
      const inicio = images.length + 1;
      for (const url of item.urls) {
        images.push(await urlToOpenAI(url, `product-${++contador}.jpg`));
      }
      produtos.push({
        descricao: descricaoDoProduto(item),
        comoVestir: comoVestirProduto(item),
        inicio,
        fim: images.length,
      });
    }
    if (glassesFiles.length) {
      const inicio = images.length + 1;
      for (const f of glassesFiles) {
        images.push(await fileToOpenAI(f.path, f.mimetype, `product-${++contador}.jpg`));
      }
      produtos.push({
        descricao: 'the product',
        comoVestir: "Place the product naturally and precisely on the model",
        inicio,
        fim: images.length,
      });
    }

    let outfitIdx = null;
    if (!keepOutfitBool && outfitFile) {
      const outfitRef = await fileToOpenAI(outfitFile.path, outfitFile.mimetype, 'outfit.jpg');
      images.push(outfitRef);
      outfitIdx = images.length;
    }

    let expressionIdx = null;
    if (expressionFile) {
      const exprPath = path.join(__dirname, 'public/expressions', expressionFile);
      if (fs.existsSync(exprPath)) {
        const exprMime = path.extname(expressionFile).toLowerCase() === '.png' ? 'image/png' : 'image/jpeg';
        images.push(await fileToOpenAI(exprPath, exprMime, 'expression.jpg'));
        expressionIdx = images.length;
      }
    }

    const prompt = buildCreativePrompt(modelos, produtos, keepOutfitBool, outfitIdx, expressionIdx, extraText);
    console.log(`[generate-creative] modelos=${modelFiles.join('+')} pose=${pose} produtos=${produtos.length} fotos=${totalProduto} (catalogo=${totalCatalogo}) keepOutfit=${keepOutfitBool} expr=${!!expressionFile} size=${imageSize}`);

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: images,
      prompt,
      quality: 'medium',
      size: imageSize,
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    res.json({ image: b64 });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Lista as referências fixas de "Corrente com Modelo" disponíveis
app.get('/api/corrente-refs', (req, res) => {
  res.json(Object.keys(CORRENTE_MODELO_REFS).map(key => ({
    key,
    thumb: '/' + CORRENTE_MODELO_REFS[key].replace(/^public\//, ''),
  })));
});

app.post('/api/generate-corrente-modelo', creativeUpload.fields([
  { name: 'chain', maxCount: 5 },
  { name: 'outfit', maxCount: 1 },
  { name: 'jacket', maxCount: 1 },
]), async (req, res) => {
  const chainFiles = req.files?.['chain'] || [];
  const outfitFile = req.files?.['outfit']?.[0];
  const jacketFile = req.files?.['jacket']?.[0];
  const uploadedPaths = [...chainFiles.map(f => f.path), outfitFile?.path, jacketFile?.path].filter(Boolean);
  try {
    const { modelFile, referenceKey, extraText, keepOutfit } = req.body;
    if (!referenceKey || !CORRENTE_MODELO_REFS[referenceKey])
      return res.status(400).json({ error: 'Selecione uma referência.' });
    if (!chainFiles.length) return res.status(400).json({ error: 'Envie ao menos uma foto da corrente.' });
    if (!modelFile)         return res.status(400).json({ error: 'Selecione um modelo.' });

    const keepOutfitBool = keepOutfit !== 'false'; // padrão: manter roupa da referência
    if (!keepOutfitBool && !outfitFile)
      return res.status(400).json({ error: 'Envie a foto da roupa ou selecione "manter roupa da referência".' });

    const modelPath = path.join(__dirname, 'public/models', modelFile);
    if (!fs.existsSync(modelPath))
      return res.status(400).json({ error: 'Modelo não encontrado.' });

    const ext  = path.extname(modelFile).toLowerCase();
    const mime = ext === '.png' ? 'image/png' : 'image/jpeg';

    const refPath = path.join(__dirname, CORRENTE_MODELO_REFS[referenceKey]);
    const referenceRef = await fileToOpenAI(refPath, 'image/jpeg', 'reference.jpg');
    const images = [referenceRef];

    const modelRef = await fileToOpenAI(modelPath, mime, 'model.jpg');
    images.push(modelRef);

    for (let i = 0; i < chainFiles.length; i++) {
      images.push(await fileToOpenAI(chainFiles[i].path, chainFiles[i].mimetype, `chain-${i + 1}.jpg`));
    }

    let outfitIdx = null;
    if (!keepOutfitBool && outfitFile) {
      const outfitRef = await fileToOpenAI(outfitFile.path, outfitFile.mimetype, 'outfit.jpg');
      images.push(outfitRef);
      outfitIdx = images.length;
    }

    let jacketIdx = null;
    const jacketMode = CORRENTE_MODELO_JACKET_MODE[referenceKey] || 'add';
    if (jacketFile) {
      const jacketRef = await fileToOpenAI(jacketFile.path, jacketFile.mimetype, 'jacket.jpg');
      images.push(jacketRef);
      jacketIdx = images.length;
    }

    const prompt = buildCorrenteModeloPrompt(chainFiles.length, keepOutfitBool, outfitIdx, extraText, jacketIdx, jacketMode);
    console.log(`[generate-corrente-modelo] ref=${referenceKey} model=${modelFile} chains=${chainFiles.length} keepOutfit=${keepOutfitBool} jacket=${!!jacketFile}(${jacketMode})`);

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: images,
      prompt,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    res.json({ image: b64 });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Lista as referências fixas de "Bracelete com Modelo" disponíveis
app.get('/api/bracelete-refs', (req, res) => {
  res.json(Object.keys(BRACELETE_MODELO_REFS).map(key => ({
    key,
    thumb: '/' + BRACELETE_MODELO_REFS[key].replace(/^public\//, ''),
    visible: BRACELETE_MODELO_VISIBLE[key] || { shirt: true, pants: true },
  })));
});

app.post('/api/generate-bracelete-modelo', creativeUpload.fields([
  { name: 'bracelet', maxCount: 5 },
  { name: 'shirt', maxCount: 1 },
  { name: 'pants', maxCount: 1 },
]), async (req, res) => {
  const braceletFiles = req.files?.['bracelet'] || [];
  const shirtFile  = req.files?.['shirt']?.[0];
  const pantsFile  = req.files?.['pants']?.[0];
  const uploadedPaths = [...braceletFiles.map(f => f.path), shirtFile?.path, pantsFile?.path].filter(Boolean);
  try {
    const { modelFile, referenceKey, extraText } = req.body;
    if (!referenceKey || !BRACELETE_MODELO_REFS[referenceKey])
      return res.status(400).json({ error: 'Selecione uma referência.' });
    if (!braceletFiles.length) return res.status(400).json({ error: 'Envie ao menos uma foto do bracelete.' });
    if (!modelFile)            return res.status(400).json({ error: 'Selecione um modelo.' });

    const modelPath = path.join(__dirname, 'public/models', modelFile);
    if (!fs.existsSync(modelPath))
      return res.status(400).json({ error: 'Modelo não encontrado.' });

    const ext  = path.extname(modelFile).toLowerCase();
    const mime = ext === '.png' ? 'image/png' : 'image/jpeg';

    const refPath = path.join(__dirname, BRACELETE_MODELO_REFS[referenceKey]);
    const referenceRef = await fileToOpenAI(refPath, 'image/jpeg', 'reference.jpg');
    const images = [referenceRef];

    const modelRef = await fileToOpenAI(modelPath, mime, 'model.jpg');
    images.push(modelRef);

    for (let i = 0; i < braceletFiles.length; i++) {
      images.push(await fileToOpenAI(braceletFiles[i].path, braceletFiles[i].mimetype, `bracelet-${i + 1}.jpg`));
    }

    let shirtIdx = null;
    if (shirtFile) {
      images.push(await fileToOpenAI(shirtFile.path, shirtFile.mimetype, 'shirt.jpg'));
      shirtIdx = images.length;
    }

    let pantsIdx = null;
    if (pantsFile) {
      images.push(await fileToOpenAI(pantsFile.path, pantsFile.mimetype, 'pants.jpg'));
      pantsIdx = images.length;
    }

    const prompt = buildBraceleteModeloPrompt(braceletFiles.length, shirtIdx, pantsIdx, extraText);
    console.log(`[generate-bracelete-modelo] ref=${referenceKey} model=${modelFile} bracelets=${braceletFiles.length} shirt=${!!shirtFile} pants=${!!pantsFile}`);

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: images,
      prompt,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    res.json({ image: b64 });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Lista as referências fixas de "Anel com Modelo" disponíveis
app.get('/api/anel-refs', (req, res) => {
  res.json(Object.keys(ANEL_MODELO_REFS).map(key => ({
    key,
    thumb: '/' + ANEL_MODELO_REFS[key].replace(/^public\//, ''),
    visible: ANEL_MODELO_VISIBLE[key] || { shirt: true, pants: true },
  })));
});

app.post('/api/generate-anel-modelo', creativeUpload.fields([
  { name: 'ring', maxCount: 5 },
  { name: 'shirt', maxCount: 1 },
  { name: 'pants', maxCount: 1 },
]), async (req, res) => {
  const ringFiles = req.files?.['ring'] || [];
  const shirtFile = req.files?.['shirt']?.[0];
  const pantsFile = req.files?.['pants']?.[0];
  const uploadedPaths = [...ringFiles.map(f => f.path), shirtFile?.path, pantsFile?.path].filter(Boolean);
  try {
    const { modelFile, referenceKey, extraText } = req.body;
    if (!referenceKey || !ANEL_MODELO_REFS[referenceKey])
      return res.status(400).json({ error: 'Selecione uma referência.' });
    if (!ringFiles.length) return res.status(400).json({ error: 'Envie ao menos uma foto do anel.' });
    if (!modelFile)        return res.status(400).json({ error: 'Selecione um modelo.' });

    const modelPath = path.join(__dirname, 'public/models', modelFile);
    if (!fs.existsSync(modelPath))
      return res.status(400).json({ error: 'Modelo não encontrado.' });

    const ext  = path.extname(modelFile).toLowerCase();
    const mime = ext === '.png' ? 'image/png' : 'image/jpeg';

    const refPath = path.join(__dirname, ANEL_MODELO_REFS[referenceKey]);
    const referenceRef = await fileToOpenAI(refPath, 'image/jpeg', 'reference.jpg');
    const images = [referenceRef];

    const modelRef = await fileToOpenAI(modelPath, mime, 'model.jpg');
    images.push(modelRef);

    for (let i = 0; i < ringFiles.length; i++) {
      images.push(await fileToOpenAI(ringFiles[i].path, ringFiles[i].mimetype, `ring-${i + 1}.jpg`));
    }

    let shirtIdx = null;
    if (shirtFile) {
      images.push(await fileToOpenAI(shirtFile.path, shirtFile.mimetype, 'shirt.jpg'));
      shirtIdx = images.length;
    }

    let pantsIdx = null;
    if (pantsFile) {
      images.push(await fileToOpenAI(pantsFile.path, pantsFile.mimetype, 'pants.jpg'));
      pantsIdx = images.length;
    }

    const prompt = buildAnelModeloPrompt(ringFiles.length, shirtIdx, pantsIdx, extraText);
    console.log(`[generate-anel-modelo] ref=${referenceKey} model=${modelFile} rings=${ringFiles.length} shirt=${!!shirtFile} pants=${!!pantsFile}`);

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: images,
      prompt,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    res.json({ image: b64 });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Seção Livre: o prompt é escrito por você, sem molde nenhum por cima.
// Com foto anexada usa images.edit; sem foto, images.generate (cria do zero).
const MAX_FOTOS_LIVRE = 10;

app.post('/api/generate-livre', creativeUpload.array('images', MAX_FOTOS_LIVRE), async (req, res) => {
  const arquivos = req.files || [];
  const uploadedPaths = arquivos.map(f => f.path);
  try {
    const { prompt, size } = req.body;
    if (!prompt || !prompt.trim()) return res.status(400).json({ error: 'Escreva o prompt.' });

    // Fotos do catálogo da loja também valem aqui
    let catalogUrls = [];
    if (req.body.catalogUrls) {
      try {
        catalogUrls = JSON.parse(req.body.catalogUrls);
        if (!Array.isArray(catalogUrls)) throw new Error();
      } catch {
        return res.status(400).json({ error: 'catalogUrls inválido.' });
      }
    }
    if (arquivos.length + catalogUrls.length > MAX_FOTOS_LIVRE)
      return res.status(400).json({ error: `No máximo ${MAX_FOTOS_LIVRE} fotos.` });

    const allowedSizes = ['1024x1024', '1024x1536', '1536x1024'];
    const imageSize = allowedSizes.includes(size) ? size : '1024x1024';

    const images = [];
    for (let i = 0; i < catalogUrls.length; i++) {
      images.push(await urlToOpenAI(catalogUrls[i], `catalogo-${i + 1}.jpg`));
    }
    for (let i = 0; i < arquivos.length; i++) {
      images.push(await fileToOpenAI(arquivos[i].path, arquivos[i].mimetype, `imagem-${i + 1}.jpg`));
    }

    console.log(`[generate-livre] fotos=${images.length} size=${imageSize} prompt="${prompt.trim().slice(0, 80)}"`);
    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    const response = images.length
      ? await client.images.edit({ model: 'gpt-image-2', image: images, prompt: prompt.trim(), quality: 'medium', size: imageSize })
      : await client.images.generate({ model: 'gpt-image-2', prompt: prompt.trim(), quality: 'medium', size: imageSize });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    res.json({ image: b64 });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3333;
app.listen(PORT, () => console.log(`Vyser rodando em http://localhost:${PORT}`));
