// Render .mmd -> .svg (mmdc), add start arrowheads to undirected "---" links
// (used to lay the decoder out upwards), then screenshot the SVG to PNG.
//
// Usage (from this folder):
//   npm i --prefix . @mermaid-js/mermaid-cli@11   # v12 changes the layout
//   node render.mjs unetControlnet 2    # name without extension, pixel scale
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import puppeteer from 'puppeteer';

const [name, scale = '2'] = process.argv.slice(2);
execFileSync('./node_modules/.bin/mmdc', ['-i', `${name}.mmd`, '-o', `${name}.svg`, '-b', 'white']);
const src = readFileSync(`${name}.mmd`, 'utf8');
const undirected = [...src.matchAll(/^\s*(\w+)(?:\[[^\]]*\])?\s+-{3,}\s+(\w+)\s*$/gm)].map(m => `L_${m[1]}_${m[2]}_0`);
let svg = readFileSync(`${name}.svg`, 'utf8');
for (const id of undirected) {
  const re = new RegExp(`(<path[^>]*data-id="${id}")`);
  if (!re.test(svg)) throw new Error(`edge ${id} not found`);
  svg = svg.replace(re, `$1 marker-start="url(#my-svg_flowchart-v2-pointStart)"`);
}
writeFileSync(`${name}.svg`, svg);

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
await page.setViewport({ width: 4000, height: 4000, deviceScaleFactor: Number(scale) });
await page.setContent(`<html><body style="margin:0;background:white">${svg}</body></html>`);
const el = await page.$('svg');
await el.screenshot({ path: `${name}.png`, omitBackground: false });
await browser.close();
console.log('ok', undirected);
