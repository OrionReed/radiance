import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

// Every folder under site/ with an index.html is one page, served at /<folder>/.
// Folders starting with _ or . are not pages (shared code, templates).
const siteDir = resolve(import.meta.dirname, 'site');

interface Page {
  name: string;
  title: string;
  description: string;
  section: string; // from <meta name="section">; '' for the main list
  order: number; // from <meta name="order">; pages without one come after, by name
}

function findPages(): Page[] {
  return readdirSync(siteDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !/^[_.]/.test(d.name) && existsSync(resolve(siteDir, d.name, 'index.html')))
    .map((d) => {
      const html = readFileSync(resolve(siteDir, d.name, 'index.html'), 'utf8');
      const title = html.match(/<title>([^<]*)<\/title>/)?.[1] ?? d.name;
      const description = html.match(/<meta\s+name="description"\s+content="([^"]*)"/)?.[1] ?? '';
      const section = html.match(/<meta\s+name="section"\s+content="([^"]*)"/)?.[1] ?? '';
      const order = Number(html.match(/<meta\s+name="order"\s+content="([^"]*)"/)?.[1] ?? Infinity);
      return { name: d.name, title, description, section, order };
    })
    .sort((a, b) => a.order - b.order || a.name.localeCompare(b.name));
}

const escape = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

// Fills <!-- pages --> in site/index.html with a link per page that names no section, and <!-- pages NAME --> with the
// pages whose <meta name="section"> is NAME.
function pageIndex(): Plugin {
  return {
    name: 'page-index',
    transformIndexHtml: {
      order: 'pre',
      handler(html, ctx) {
        if (resolve(ctx.filename) !== resolve(siteDir, 'index.html')) return html;
        const pages = findPages();
        return html.replace(/<!-- pages(?: ([\w-]+))? -->/g, (_, section = '') => {
          const items = pages
            .filter((p) => p.section === section)
            .map(
              (p) =>
                `<li><a href="./${p.name}/">${escape(p.title)}</a>` +
                (p.description ? `<span>${escape(p.description)}</span>` : '') +
                `</li>`,
            )
            .join('\n');
          return items || '<li class="empty">No pages yet.</li>';
        });
      },
    },
  };
}

export default defineConfig({
  root: siteDir,
  // Relative base, so the build works under any GitHub Pages path (user.github.io/<repo>/).
  base: './',
  plugins: [pageIndex()],
  build: {
    outDir: resolve(import.meta.dirname, 'dist'),
    emptyOutDir: true,
    target: 'esnext',
    rollupOptions: {
      input: {
        index: resolve(siteDir, 'index.html'),
        ...Object.fromEntries(findPages().map((p) => [p.name, resolve(siteDir, p.name, 'index.html')])),
      },
    },
  },
});
