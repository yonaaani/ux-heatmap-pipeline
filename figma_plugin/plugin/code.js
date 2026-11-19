// UX-аналіз макета (локально) — частина плагіна, що працює всередині Figma (доступ до шарів).
// 1) Виділений фрейм → структура для пайплайну (тип, рамка, кольори, шрифти з шарів; крок 1 не потрібен) + PNG 1×.
// 2) Виправлення з плану (крок 8) → зміни шарів (кнопка «Застосувати»; усе скасовується Ctrl+Z).
// 3) Картинки результатів (кроки 2–7) → новий фрейм «UX-аналіз · …» праворуч від макета.
// Аналіз робить локальний сервер (figma_plugin/server.py); мережеві запити — з ui.html.

var state = { frameId: null, labelOf: {} };

// ------------------------------------------------------------------ кольори й шрифти

function toHex(c) {
  function h(v) { var s = Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).toUpperCase(); return s.length < 2 ? "0" + s : s; }
  return "#" + h(c.r) + h(c.g) + h(c.b);
}

function fromHex(hex) {
  var n = parseInt(hex.replace("#", ""), 16);
  return { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255 };
}

function paintsOf(node, key) {
  var p = node[key || "fills"];
  if (!p || typeof p === "symbol" || !p.length) return [];
  return p.filter(function (x) { return x.visible !== false && (x.opacity === undefined || x.opacity > 0.05); });
}

function solidHex(node, key) {  // верхня (остання) суцільна заливка
  var p = paintsOf(node, key).filter(function (x) { return x.type === "SOLID"; });
  return p.length ? toHex(p[p.length - 1].color) : null;
}

function hasImageFill(node) {
  return paintsOf(node).some(function (x) { return x.type === "IMAGE" || x.type === "VIDEO"; });
}

function fontSizeOf(t) {
  var s = t.fontSize;
  if (typeof s !== "number") s = t.getRangeFontSize(0, 1);
  return typeof s === "number" ? s : 16;
}

function isBold(t) {
  var w = t.fontWeight;
  if (typeof w !== "number" && t.getRangeFontWeight) w = t.getRangeFontWeight(0, 1);
  if (typeof w === "number") return w >= 600;
  var fn = typeof t.fontName === "symbol" ? t.getRangeFontName(0, 1) : t.fontName;
  return /bold|semi|black|heavy/i.test((fn && fn.style) || "");
}

function saturation(hex) {
  var c = fromHex(hex), mx = Math.max(c.r, c.g, c.b), mn = Math.min(c.r, c.g, c.b);
  return mx === 0 ? 0 : (mx - mn) / mx;
}

function colorDist(a, b) {
  var x = fromHex(a), y = fromHex(b);
  return Math.abs(x.r - y.r) + Math.abs(x.g - y.g) + Math.abs(x.b - y.b);
}

// ------------------------------------------------------------------ фрейм → структура для пайплайну

var PRICE_RE = /(\d[\d\s.,]*\s*(грн|₴|uah|usd|eur|\$|€|руб))|([$€₴£]\s?\d)/i;
var DECOR_RE = /decor|декор|promo|промо|banner|банер|sale|badge|бейдж/;
// текст-підказка поля вводу («What are you looking for?», «Enter your email», «Пошук…»)
var PLACEHOLDER_RE = /^(search|enter |type |your |what are you looking|find |пошук|шукати|введіть|ваш[аіе]? |e-?mail$|password$|пароль$)/i;
var KEEP_RE = /keep|brand|бренд|захищ|не змін/i;  // захищені шари: плагін їх не змінює й не ховає

function isProtected(node, stopAt) {  // сам шар або будь-який з батьків (до фрейму) має keep / бренд у назві
  for (var n = node; n && n !== stopAt && n.type !== "PAGE"; n = n.parent) if (KEEP_RE.test(n.name || "")) return true;
  return false;
}
var VECTOR_TYPES = ["VECTOR", "BOOLEAN_OPERATION", "STAR", "POLYGON", "ELLIPSE"];

function textColor(t) {  // колір тексту; якщо кольорів кілька — той, яким написано найбільше символів
  var c = solidHex(t);
  if (c) return c;
  try {
    var count = {};
    t.getStyledTextSegments(["fills"]).forEach(function (s) {
      var p = (s.fills || []).filter(function (x) { return x.type === "SOLID" && x.visible !== false; });
      if (p.length) { var h = toHex(p[p.length - 1].color); count[h] = (count[h] || 0) + s.characters.replace(/\s/g, "").length; }
    });
    var best = null;
    Object.keys(count).forEach(function (h) { if (!best || count[h] > count[best]) best = h; });
    return best;
  } catch (e) { return null; }
}

function vectorOnly(node) {  // група лише з векторних фігур (логотип, ілюстрація, текст у кривих): кількість фігур або -1
  var leaves = 0, ok = true;
  (function rec(n, d) {
    if (!ok || n.visible === false) return;
    if ("children" in n && n.type !== "BOOLEAN_OPERATION") { if (d > 8) { ok = false; return; } n.children.forEach(function (c) { rec(c, d + 1); }); return; }
    if (n.type === "TEXT" || hasImageFill(n)) { ok = false; return; }
    leaves++;
  })(node, 0);
  return ok ? leaves : -1;
}

function textsInside(node, limit) {
  var out = [];
  (function rec(n, d) {
    if (!("children" in n) || d > 4) return;
    for (var i = 0; i < n.children.length && out.length < (limit || 50); i++) {
      var c = n.children[i];
      if (c.visible === false) continue;
      if (c.type === "TEXT" && c.characters.trim()) out.push(c); else rec(c, d + 1);
    }
  })(node, 0);
  return out;
}

function frameToLayout(frame) {
  var F = frame.absoluteBoundingBox;
  var W = F.width, H = F.height;
  var background = solidHex(frame) || "#FFFFFF";
  var elements = [], labelOf = {}, consumed = {};

  function boxOf(n) {
    var b = n.absoluteBoundingBox;
    return b ? [b.x - F.x, b.y - F.y, b.width, b.height] : null;
  }

  function textElement(t, box, type) {
    var size = fontSizeOf(t), bold = isBold(t), chars = t.characters.replace(/\s+/g, " ").trim();
    // заголовок: великий кегль або жирний короткий рядок; довгий абзац 20 px (вступний текст) — це текст
    if (!type) type = PRICE_RE.test(chars) && chars.length <= 24 ? "price"
      : (size >= 24 || (size >= 20 && chars.length <= 80) || (bold && size >= 17 && chars.length <= 80)) ? "heading" : "text";
    var al = typeof t.textAlignHorizontal === "string" ? t.textAlignHorizontal.toLowerCase() : "left";
    return { id: t.id, type: type, role: type === "text" ? "secondary" : "primary", box: box, text: chars,
             color: textColor(t) || "#222222", font_size: Math.round(size * 10) / 10, bold: bold,
             align: al === "justified" ? "left" : al };
  }

  function control(n, box, type, texts) {  // кнопка / чип / поле: текст підпису — з шару тексту всередині
    var e = { id: n.id, type: type, role: type === "button" ? "primary" : "secondary", box: box,
              fill: solidHex(n), radius: typeof n.cornerRadius === "number" ? n.cornerRadius : 0 };
    if (!e.fill && type !== "input") e.fill = null;
    if (texts.length) {
      var t = texts[0];
      e.text = texts.map(function (x) { return x.characters; }).join(" ").replace(/\s+/g, " ").trim();
      e.color = textColor(t) || "#222222";
      e.font_size = Math.round(fontSizeOf(t) * 10) / 10;
      e.bold = isBold(t);
      if (typeof t.textAlignHorizontal === "string") e.align = t.textAlignHorizontal === "JUSTIFIED" ? "left" : t.textAlignHorizontal.toLowerCase();
      labelOf[n.id] = t.id;
      texts.forEach(function (x) { consumed[x.id] = true; });
    }
    return e;
  }

  function classify(n, box, hint) {
    var w = box[2], h = box[3], fill = solidHex(n), stroke = solidHex(n, "strokes");
    if (VECTOR_TYPES.indexOf(n.type) >= 0 && w <= 8 && h <= 8 && n.type !== "LINE") return null;  // крихта (літера логотипа в кривих, точка) — не ціль для пальця
    if (n.type === "LINE" || ((fill || stroke) && (h <= 2 || w <= 2))) return "divider";
    if (/(^|[^a-z])cta([^a-z]|$)|головн|primary.?button/.test(hint)) return "cta";
    if (VECTOR_TYPES.indexOf(n.type) >= 0 && w <= 64 && h <= 64) return "icon";  // векторна фігура — іконка, як би не звалась
    if (/button|btn|кнопк/.test(hint)) return "button";
    if (/chip|tag|badge|pill|тег|чіп|бейдж/.test(hint)) return "chip";
    if (/input|field|search|textbox|поле|пошук/.test(hint)) return "input";
    if (hasImageFill(n)) return /avatar|face|аватар|обличч/.test(hint) ? "face" : (w <= 48 && h <= 48 ? "icon" : "image");  // дрібна картинка — іконка
    if (/icon|іконк|ico[_\-\s]/.test(hint) || (VECTOR_TYPES.indexOf(n.type) >= 0 && w <= 64 && h <= 64)) return "icon";
    var texts = textsInside(n, 3);
    // логотип / ілюстрація з векторних фігур — одна картинка, а не десятки «іконок»
    // (без власної заливки або з назвою logo; не ряд іконок — тоді кілька вкладених груп)
    if ("children" in n && !texts.length && (w > 48 || h > 48 || /logo|лого/.test(hint)) && (!fill || /logo|лого|illustr|ілюстр/.test(hint)) &&
        n.children.filter(function (c) { return "children" in c && c.type !== "BOOLEAN_OPERATION"; }).length < 2 && vectorOnly(n) >= 2) return "image";
    if ((fill || stroke) && texts.length === 1 && h >= 20 && h <= 72 && w <= 0.95 * W) {  // мобільна кнопка на всю ширину з полями
      var t0 = texts[0].characters.trim();
      if (PLACEHOLDER_RE.test(t0)) return "input";         // поле пошуку / e-mail з підказкою
      if (h <= 26 && fontSizeOf(texts[0]) <= 14) return "chip";  // маленька мітка «Sale», «New», «Lesson 1»
      return "button";
    }
    if (w <= 48 && h <= 48 && !texts.length && (fill || VECTOR_TYPES.indexOf(n.type) >= 0 || "children" in n)) return "icon";
    if (fill) return "card";
    return null;
  }

  var nodeById = {};
  function walk(n) {
    nodeById[n.id] = n;
    if (n.visible === false || (typeof n.opacity === "number" && n.opacity < 0.05)) return;
    var box = boxOf(n);
    if (!box || box[0] >= W || box[1] >= H || box[0] + box[2] <= 0 || box[1] + box[3] <= 0) return;
    if (n.type === "TEXT") {
      if (!consumed[n.id] && n.characters.trim()) elements.push(textElement(n, box));
      return;
    }
    var hint = (n.name || "").toLowerCase();
    var kind = classify(n, box, hint), descend = true;
    // шар «cta» — цілий блок (кнопка + підпис під нею, заголовок над нею): шукаємо саму кнопку всередині
    if (kind === "cta" && textsInside(n, 5).length > 1) {
      ctaScopes.push({ node: n, box: box, start: elements.length });
      if (solidHex(n) && colorDist(solidHex(n), background) > 0.03) elements.push({ id: n.id, type: "card", role: "secondary", box: box, fill: solidHex(n),
                                       stroke: solidHex(n, "strokes"), radius: typeof n.cornerRadius === "number" ? n.cornerRadius : 0 });
      if ("children" in n) n.children.forEach(walk);
      ctaScopes[ctaScopes.length - 1].end = elements.length;
      return;
    }
    if (kind === "cta" || kind === "button" || kind === "chip" || kind === "input") {
      var e = control(n, box, kind === "cta" ? "button" : kind, textsInside(n, 5));
      if (kind === "cta") e.role = "cta";
      elements.push(e);
      descend = false;
    } else if (kind === "icon") {
      var ic = solidHex(n);
      if (!ic && "children" in n) for (var i = 0; i < n.children.length && !ic; i++) ic = solidHex(n.children[i]);
      elements.push({ id: n.id, type: "icon", role: "decor", box: box, fill: ic || null });
      descend = false;
    } else if (kind === "image" || kind === "face") {
      elements.push({ id: n.id, type: kind, role: "decor", box: box, fill: solidHex(n) || "#CCCCCC",
                      radius: typeof n.cornerRadius === "number" ? n.cornerRadius : 0 });
      // фото-фон з текстом і кнопками поверх (банер, hero): вміст аналізуємо; логотип чи звичайна картинка — ні
      descend = hasImageFill(n) && textsInside(n, 1).length > 0;
    } else if (kind === "divider") {
      elements.push({ id: n.id, type: "divider", role: "decor", box: box, fill: solidHex(n) || solidHex(n, "strokes") });
      descend = false;
    } else if (kind === "card") {
      elements.push({ id: n.id, type: "card", role: "secondary", box: box, fill: solidHex(n),
                      stroke: solidHex(n, "strokes"), radius: typeof n.cornerRadius === "number" ? n.cornerRadius : 0 });
    }
    if (descend && "children" in n) n.children.forEach(walk);
  }

  var ctaScopes = [];
  frame.children.forEach(walk);

  function inside(a, b, pad) {  // рамка a всередині рамки b
    return a[0] >= b[0] - pad && a[1] >= b[1] - pad && a[0] + a[2] <= b[0] + b[2] + pad && a[1] + a[3] <= b[1] + b[3] + pad;
  }
  var TEXTY = ["text", "heading", "price"];

  // прямокутник і текст поверх нього — сусідні шари, а не вкладені (типово для макетів «з руки»):
  // це кнопка (текст по центру) або поле вводу (світле поле, текст ліворуч / сірий підказковий)
  elements.slice().forEach(function (c) {
    if (c.type !== "card" || c.box[3] < 20 || c.box[3] > 72 || c.box[2] > 0.95 * W) return;
    var i0 = elements.indexOf(c);
    var tx = elements.filter(function (t, i) { return i > i0 && TEXTY.indexOf(t.type) >= 0 && inside(t.box, c.box, 2); });
    var others = elements.filter(function (t, i) {
      return i > i0 && t !== c && tx.indexOf(t) < 0 && t.type !== "icon" && inside(t.box, c.box, 2);
    });
    if (tx.length !== 1 || others.length) return;
    var t = tx[0], tcx = t.box[0] + t.box[2] / 2, ccx = c.box[0] + c.box[2] / 2;
    var centered = Math.abs(tcx - ccx) <= 0.12 * c.box[2];
    var fill = c.fill || background, light = colorDist(fill, "#FFFFFF") < 0.15;
    var grey = t.color && saturation(t.color) < 0.12 && colorDist(t.color, "#000000") > 0.35 && colorDist(t.color, "#FFFFFF") > 0.2;
    var type = (light && (!centered || grey)) || PLACEHOLDER_RE.test(t.text) ? "input"
      : (c.box[3] <= 26 && (t.font_size || 16) <= 14) ? "chip" : "button";
    c.type = type; c.role = type === "button" ? "primary" : "secondary";
    elements.forEach(function (ic) { if (ic.type === "icon" && inside(ic.box, c.box, 2)) ic.role = "secondary"; });  // значок у полі / кнопці — не декор
    c.text = t.text; c.color = t.color; c.font_size = t.font_size; c.bold = t.bold; c.align = t.align;
    labelOf[c.id] = t.id;
    elements.splice(elements.indexOf(t), 1);
  });

  // блок «cta»: головна кнопка — найбільша кнопка всередині; якщо кнопки немає — весь блок як раніше
  ctaScopes.forEach(function (s) {
    var btn = null;
    elements.forEach(function (e) {
      if (e.type === "button" && e.id !== s.node.id && inside(e.box, s.box, 2) && (!btn || e.box[2] * e.box[3] > btn.box[2] * btn.box[3])) btn = e;
    });
    if (btn) { btn.role = "cta"; return; }
    var e = control(s.node, s.box, "button", textsInside(s.node, 5));
    e.role = "cta";
    elements = elements.filter(function (x) { return x.id === s.node.id ? false : !consumed[x.id] || TEXTY.indexOf(x.type) < 0; });
    elements.push(e);
  });

  // заливка, повністю закрита фото над нею (кольорова підкладка під картинкою), — не видно, не аналізуємо
  elements = elements.filter(function (c, i) {
    if (c.type !== "card") return true;
    return !elements.some(function (m, j) {
      return j > i && (m.type === "image" || m.type === "face") && inside(c.box, m.box, 1);
    });
  });

  // назви шарів (для повідомлень) і декоративні шари: слово decor / декор / promo / banner / sale / badge у назві
  elements.forEach(function (e) {
    var node = nodeById[e.id];
    e.name = node ? node.name : e.id;
    // стандартна назва Figma («Frame 161», «Rectangle 82») нічого не каже і повторюється — для кнопки беремо її підпис
    if (e.text && /^(frame|group|rectangle|vector|instance|component|auto layout)\s*\d*$/i.test((e.name || "").trim())) e.name = e.text;
    if (e.role !== "cta" && DECOR_RE.test((e.name || "").toLowerCase())) e.role = "decor";
    if (node && isProtected(node, frame)) e.keep = true;
  });

  // іконка поруч із текстом (годинник біля «30 хв», значок біля пункту) несе зміст — це не декор
  elements.forEach(function (e) {
    if (e.type !== "icon" || e.role !== "decor" || DECOR_RE.test((e.name || "").toLowerCase())) return;
    var cy = e.box[1] + e.box[3] / 2;
    var near = elements.some(function (t) {
      if (["text", "heading", "price"].indexOf(t.type) < 0) return false;
      var ty = t.box[1] + t.box[3] / 2, gap = Math.max(t.box[0] - (e.box[0] + e.box[2]), e.box[0] - (t.box[0] + t.box[2]));
      return Math.abs(ty - cy) <= Math.max(e.box[3], t.box[3]) * 0.6 && gap >= -2 && gap <= 24;
    });
    if (near) e.role = "secondary";
  });

  // кнопка дії (CTA): шар з назвою «cta»; інакше — найбільша насичена кнопка, помітна на фоні
  if (!elements.some(function (e) { return e.role === "cta"; })) {
    var best = null, bestScore = 0;
    elements.forEach(function (e) {
      if (e.type !== "button") return;
      var main = e.fill || e.color || background;
      var score = e.box[2] * e.box[3] * (0.3 + saturation(main)) * (colorDist(main, background) > 0.15 ? 1 : 0.2);
      if (score > bestScore) { best = e; bestScore = score; }
    });
    if (best) best.role = "cta";
  }
  return { layout: { width: Math.round(W), height: Math.round(H), background: background, elements: elements },
           labelOf: labelOf };
}

// ------------------------------------------------------------------ виправлення кроку 8 → шари

async function getNode(id) {
  return figma.getNodeByIdAsync ? await figma.getNodeByIdAsync(id) : figma.getNodeById(id);
}

function setSolid(node, hex) {
  var old = paintsOf(node).filter(function (x) { return x.type === "SOLID"; });
  var op = old.length && old[old.length - 1].opacity !== undefined ? old[old.length - 1].opacity : 1;
  node.fills = [{ type: "SOLID", color: fromHex(hex), opacity: op }];
}

async function loadFontsOf(t) {
  var names = typeof t.fontName === "symbol" ? t.getRangeAllFontNames(0, t.characters.length) : [t.fontName];
  for (var i = 0; i < names.length; i++) await figma.loadFontAsync(names[i]);
}

async function labelNode(node) {
  if (node.type === "TEXT") return node;
  if (state.labelOf[node.id]) { var l = await getNode(state.labelOf[node.id]); if (l) return l; }
  var t = textsInside(node, 1);
  return t.length ? t[0] : null;
}

async function applyFix(fix, frame) {
  var log = [];
  if (fix.remove) {
    for (var i = 0; i < fix.remove.length; i++) {
      var r = await getNode(fix.remove[i]);
      if (r && isProtected(r, frame)) { log.push("«" + r.name + "»: шар захищено, не сховано"); continue; }
      if (r) { r.visible = false; log.push("сховано «" + r.name + "»"); }
    }
    return log;
  }
  var node = await getNode(fix.id);
  if (!node) return ["шар " + fix.id + " не знайдено"];
  if (isProtected(node, frame)) return ["«" + node.name + "»: шар захищено (keep / бренд), не змінено"];
  var set = fix.set || {};
  for (var field in set) {
    var v = set[field];
    try {
      if (field === "fill" && (hasImageFill(node) || paintsOf(node).some(function (p) { return p.type.indexOf("GRADIENT") === 0; }))) {
        log.push("«" + node.name + "»: фото чи градієнт — колір " + v + " лише порада, шар не змінено");
      } else if (field === "fill") {
        setSolid(node, v); log.push("«" + node.name + "»: заливка " + v);
      } else if (field === "color") {
        var t = await labelNode(node);
        if (t) { setSolid(t, v); log.push("«" + t.name + "»: колір тексту " + v); }
      } else if (field === "font_size" || field === "bold") {
        var t2 = await labelNode(node);
        if (!t2) continue;
        await loadFontsOf(t2);
        if (field === "font_size") { t2.fontSize = v; log.push("«" + t2.name + "»: шрифт " + v + " px"); }
        else if (v) {
          var fam = (typeof t2.fontName === "symbol" ? t2.getRangeFontName(0, 1) : t2.fontName).family, done = false;
          var styles = ["Bold", "SemiBold", "Semi Bold", "Semibold"];
          for (var s = 0; s < styles.length && !done; s++) {
            try { await figma.loadFontAsync({ family: fam, style: styles[s] }); t2.fontName = { family: fam, style: styles[s] }; done = true; }
            catch (e) { /* такого накреслення немає — пробуємо наступне */ }
          }
          log.push(done ? "«" + t2.name + "»: жирний шрифт" : "«" + t2.name + "»: жирного накреслення " + fam + " немає");
        }
      } else if (field === "box") {
        var F = frame.absoluteBoundingBox, cur = node.absoluteBoundingBox;
        var inAuto = node.parent && node.parent.layoutMode && node.parent.layoutMode !== "NONE" && node.layoutPositioning !== "ABSOLUTE";
        if ("resize" in node) node.resize(Math.max(1, v[2]), Math.max(1, v[3]));
        if (!inAuto) { node.x += (F.x + v[0]) - cur.x; node.y += (F.y + v[1]) - cur.y; }
        log.push("«" + node.name + "»: розмір " + Math.round(v[2]) + "×" + Math.round(v[3]) + (inAuto ? " (позицію задає auto-layout)" : ""));
      } else {
        log.push("«" + node.name + "»: " + field + " — лише порада, у Figma не застосовано");
      }
    } catch (e) {
      log.push("«" + node.name + "»: " + field + " — помилка: " + e.message);
    }
  }
  return log;
}

// ------------------------------------------------------------------ шари результатів

async function loadUiFont() {
  var opts = [{ family: "Inter", style: "Regular" }, { family: "Roboto", style: "Regular" }, { family: "Arial", style: "Regular" }];
  for (var i = 0; i < opts.length; i++) { try { await figma.loadFontAsync(opts[i]); return opts[i]; } catch (e) { /* далі */ } }
  return null;
}

async function drawLayers(frame, overlays, summary) {
  var font = await loadUiFont();
  var F = frame.absoluteBoundingBox, w = F.width, h = F.height, gap = 40, cap = font ? 36 : 0;
  var perRow = w > h ? 2 : 4;
  var rows = Math.ceil(overlays.length / perRow);
  var out = figma.createFrame();
  out.name = "UX-аналіз · " + frame.name;
  var head = font ? 140 : 0;
  out.resize(gap + perRow * (w + gap), head + rows * (h + cap + gap) + gap);
  out.x = F.x + w + 120; out.y = F.y;
  out.fills = [{ type: "SOLID", color: fromHex("#F6F6F8") }];
  if (font) {
    var t = figma.createText(); t.fontName = font; t.fontSize = 18; t.characters = summary;
    t.x = gap; t.y = 24; t.resize(out.width - 2 * gap, 100); t.textAutoResize = "HEIGHT"; out.appendChild(t);
  }
  for (var i = 0; i < overlays.length; i++) {
    var col = i % perRow, row = Math.floor(i / perRow);
    var x = gap + col * (w + gap), y = head + gap + row * (h + cap + gap);
    if (font) {
      var lab = figma.createText(); lab.fontName = font; lab.fontSize = 16; lab.characters = overlays[i].name;
      lab.x = x; lab.y = y; out.appendChild(lab);
    }
    var rect = figma.createRectangle(); rect.name = overlays[i].name; rect.resize(w, h);
    var img = figma.createImage(overlays[i].bytes);
    rect.fills = [{ type: "IMAGE", imageHash: img.hash, scaleMode: "FILL" }];
    rect.x = x; rect.y = y + cap; out.appendChild(rect);
  }
  figma.viewport.scrollAndZoomIntoView([frame, out]);
  return out;
}

// ------------------------------------------------------------------ зв'язок з ui.html

function selectedFrame() {
  var s = figma.currentPage.selection;
  if (s.length !== 1) return null;
  var n = s[0];
  return ["FRAME", "COMPONENT", "INSTANCE", "SECTION"].indexOf(n.type) >= 0 ? n : null;
}

function postSelection() {
  var f = selectedFrame();
  figma.ui.postMessage({ type: "selection", name: f ? f.name : null });
}

async function onMessage(msg) {
  try {
    if (msg.type === "analyze-request") {
      var f = selectedFrame();
      if (!f) return figma.ui.postMessage({ type: "error", message: "Виділіть один фрейм (екран макета)." });
      var conv = frameToLayout(f);
      state.frameId = f.id; state.labelOf = conv.labelOf;
      var png = await f.exportAsync({ format: "PNG", constraint: { type: "SCALE", value: 1 } });
      figma.ui.postMessage({ type: "frame-data", frameName: f.name, layout: conv.layout, png: png });
    } else if (msg.type === "apply") {
      var frame = await getNode(state.frameId);
      if (!frame) return figma.ui.postMessage({ type: "error", message: "Фрейм аналізу не знайдено." });
      var log = [];
      for (var i = 0; i < msg.fixes.length; i++) log = log.concat(await applyFix(msg.fixes[i], frame));
      figma.commitUndo();
      figma.ui.postMessage({ type: "applied", log: log });
    } else if (msg.type === "draw") {
      var fr = await getNode(state.frameId);
      if (!fr) return figma.ui.postMessage({ type: "error", message: "Фрейм аналізу не знайдено." });
      await drawLayers(fr, msg.overlays, msg.summary);
      figma.ui.postMessage({ type: "drawn" });
    }
  } catch (e) {
    figma.ui.postMessage({ type: "error", message: e.message || String(e) });
  }
}

function main() {
  figma.showUI(__html__, { width: 400, height: 700, themeColors: true });
  figma.ui.onmessage = onMessage;
  figma.on("selectionchange", postSelection);
  postSelection();
}

if (typeof figma !== "undefined" && !(typeof globalThis !== "undefined" && globalThis.__UX_TEST__)) main();
