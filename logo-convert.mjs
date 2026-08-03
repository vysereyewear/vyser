import sharp from 'sharp';

const input = 'C:/Users/joaov/Downloads/vyser (3).png';
const output = 'C:/Users/joaov/Downloads/vyser-logo-transparent.png';

const { data, info } = await sharp(input)
  .ensureAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });

const { width, height, channels } = info;

let bgR = 0, bgG = 0, bgB = 0, count = 0;
const sz = 20;
for (let y = 0; y < sz; y++) {
  for (let x = 0; x < sz; x++) {
    const i = (y * width + x) * channels;
    bgR += data[i]; bgG += data[i+1]; bgB += data[i+2];
    count++;
  }
}
bgR /= count; bgG /= count; bgB /= count;
console.log('Fundo detectado:', Math.round(bgR), Math.round(bgG), Math.round(bgB));

for (let i = 0; i < data.length; i += channels) {
  const r = data[i], g = data[i+1], b = data[i+2];
  const darkness = Math.max(0, ((bgR - r) + (bgG - g) + (bgB - b)) / 3);
  const alpha = Math.min(255, Math.round(darkness * 1.2));
  data[i]   = 0;
  data[i+1] = 0;
  data[i+2] = 0;
  data[i+3] = alpha;
}

await sharp(data, { raw: { width, height, channels } })
  .png()
  .toFile(output);

console.log('Salvo em:', output);
