// Enche o guarda-roupa puxando fotos de peças de lojas Shopify.
//
// A maioria das marcas de streetwear roda em Shopify, e toda loja Shopify expõe
// /products.json publicamente — o mesmo endpoint que a gente usa pro catálogo da
// VYSER. Dá pra pegar dezenas de peças de uma vez, já recortadas em fundo limpo,
// sem sair catando imagem na mão.
//
//   node puxar-roupas.mjs <loja> <roupas|calcas> [quantas] [filtro]
//
//   node puxar-roupas.mjs exemplo-store.com roupas 20 jacket
//   node puxar-roupas.mjs exemplo-store.com calcas 15 pant
//
// O nome do arquivo sai do título do produto, que é o que vira rótulo no app.

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const [, , loja, destino = 'roupas', quantasArg = '20', filtro = ''] = process.argv;

if (!loja) {
  console.error('uso: node puxar-roupas.mjs <loja> <roupas|calcas> [quantas] [filtro]');
  process.exit(1);
}
if (!['roupas', 'calcas'].includes(destino)) {
  console.error('destino tem que ser "roupas" ou "calcas"');
  process.exit(1);
}

const quantas = parseInt(quantasArg, 10) || 20;
const pasta = path.join(__dirname, 'public', destino);
const dominio = loja.replace(/^https?:\/\//, '').replace(/\/$/, '');

// vira nome de arquivo: "Faded Cargo Pant - Black" -> "faded-cargo-pant-black"
function comoArquivo(titulo) {
  return titulo
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
}

const res = await fetch(`https://${dominio}/products.json?limit=250`);
if (!res.ok) {
  console.error(`${dominio} respondeu ${res.status} — essa loja pode não ser Shopify.`);
  process.exit(1);
}
const { products } = await res.json();

const termo = filtro.toLowerCase();
const escolhidos = products
  .filter(p => p.images?.length)
  .filter(p => !termo || `${p.title} ${p.product_type} ${p.tags?.join(' ')}`.toLowerCase().includes(termo))
  .slice(0, quantas);

await fs.mkdir(pasta, { recursive: true });

let salvos = 0;
for (const p of escolhidos) {
  // a primeira imagem costuma ser a peça sozinha, que é o que serve de referência
  const url = p.images[0].src;
  const ext = path.extname(url.split('?')[0]) || '.jpg';
  const arquivo = path.join(pasta, comoArquivo(p.title) + ext);

  try {
    const img = await fetch(url);
    if (!img.ok) continue;
    await fs.writeFile(arquivo, Buffer.from(await img.arrayBuffer()));
    salvos++;
    console.log(`  ${path.basename(arquivo)}`);
  } catch {
    console.log(`  (falhou) ${p.title}`);
  }
}

console.log(`\n${salvos} peças salvas em public/${destino}/ — de ${products.length} produtos na loja.`);
if (termo) console.log(`filtro aplicado: "${filtro}"`);
