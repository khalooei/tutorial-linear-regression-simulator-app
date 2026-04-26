/**
 * Linear regression: sample vs parameter space, selectable loss, GD with stop,
 * MathJax sidebar, light/dark theme.
 */

const RNG_SEED = 42;

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateData(n, wTrue, bTrue, noiseStd, seed) {
  const rand = mulberry32(seed);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = 2 * rand();
    const noise = noiseStd * (rand() * 2 - 1);
    xs.push(x);
    ys.push(wTrue * x + bTrue + noise);
  }
  return { xs, ys, wTrue, bTrue, _seed: seed };
}

function mse(xs, ys, w, b) {
  let s = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (w * xs[i] + b);
    s += r * r;
  }
  return s / n;
}

function gradMse(xs, ys, w, b) {
  let gw = 0;
  let gb = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (w * xs[i] + b);
    gw += -xs[i] * r;
    gb += -r;
  }
  return { gw: (2 / n) * gw, gb: (2 / n) * gb };
}

function mae(xs, ys, w, b) {
  let s = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (w * xs[i] + b);
    s += Math.abs(r);
  }
  return s / n;
}

function gradMae(xs, ys, w, b) {
  let gw = 0;
  let gb = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (w * xs[i] + b);
    const sgn = r === 0 ? 0 : r > 0 ? 1 : -1;
    gw += -xs[i] * sgn;
    gb += -sgn;
  }
  return { gw: gw / n, gb: gb / n };
}

function huberPsi(r, delta) {
  const ar = Math.abs(r);
  if (ar <= delta) return r;
  return delta * Math.sign(r);
}

function huberLoss(xs, ys, w, b, delta) {
  let s = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (w * xs[i] + b);
    const ar = Math.abs(r);
    if (ar <= delta) s += 0.5 * r * r;
    else s += delta * (ar - 0.5 * delta);
  }
  return s / n;
}

function gradHuber(xs, ys, w, b, delta) {
  let gw = 0;
  let gb = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (w * xs[i] + b);
    const psi = huberPsi(r, delta);
    gw += -xs[i] * psi;
    gb += -psi;
  }
  return { gw: gw / n, gb: gb / n };
}

function closedFormLeastSquares(xs, ys) {
  const n = xs.length;
  let sx = 0,
    sy = 0,
    sxx = 0,
    sxy = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i],
      y = ys[i];
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return { w: 0, b: sy / n };
  const w = (n * sxy - sx * sy) / denom;
  const b = (sy - w * sx) / n;
  return { w, b };
}

function residualStats(xs, ys, w, b) {
  const n = xs.length;
  let sumAbs = 0;
  let sumSq = 0;
  const rs = [];
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (w * xs[i] + b);
    rs.push(r);
    sumAbs += Math.abs(r);
    sumSq += r * r;
  }
  return {
    n,
    meanAbs: sumAbs / n,
    meanSq: sumSq / n,
    rs,
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
  };
}

const I18N = {
  fa: {
    title: "تصویرسازی بهینه‌سازی — فضای نمونه و فضای پارامتر",
    subtitle: "رگرسیون خطی، منظرهٔ تابع هزینه، گرادیان کاهشی روی سطح هزینه",
    intro_title: "ایدهٔ آموزشی",
    intro_p1:
      "در رگرسیون خطی y ≈ w x + b، «فضای نمونه» صفحهٔ (x, y) است؛ «فضای پارامتر» صفحهٔ (w, b) است. تابع هزینهٔ انتخابی روی (w, b) منظرهٔ خطا را می‌سازد؛ گرادیان کاهشی در جهت منفی گرادیان حرکت می‌کند (زیربنای اقلیدسی روی فضای پارامتر).",
    bullet_sample:
      "فضای نمونه: محورهای x و y؛ نقاط داده؛ خط برازش و باقیمانده‌ها.",
    bullet_param:
      "فضای پارامتر: محورهای w و b؛ ترازهای تابع هزینه؛ موقعیت فعلی و مسیر گرادیان.",
    bullet_loss:
      "تابع هزینه را از منوی کنترل انتخاب کنید؛ سطح رنگی سمت راست همان تابع روی (w, b) است.",
    repro_note:
      "تکرارپذیری: داده با seed ثابت؛ گام‌های GD صریح؛ جزئیات هر گام در سایدبار با فرمول.",
    controls_title: "کنترل‌ها",
    label_loss_fn: "تابع هزینه",
    label_huber_delta: "آستانهٔ Huber δ",
    label_w: "شیب w",
    label_b: "عرض از مبدأ b",
    loss_label_mse: "میانگین مربعات خطا (MSE)",
    loss_label_mae: "میانگین قدر مطلق خطا (MAE)",
    loss_label_huber: "میانگین هزینهٔ Huber",
    gd_title: "گرادیان کاهشی (تکرارپذیر)",
    gd_steps_desc:
      "گام: (w, b) ← (w, b) − η ∇L. می‌توانید انیمیشن را متوقف کنید؛ تأخیر بین گام‌ها را تنظیم کنید.",
    label_lr: "نرخ یادگیری η",
    label_steps: "تعداد گام",
    label_delay: "تأخیر انیمیشن (میلی‌ثانیه)",
    label_seed: "دادهٔ ثابت (seed=42)",
    btn_animate: "شروع انیمیشن GD",
    btn_stop: "توقف",
    btn_reset: "بازنشانی خط",
    btn_opt: "پرش به جواب حداقل مربعات (MSE)",
    btn_log_state: "ثبت وضعیت فعلی در جزئیات",
    btn_clear_log: "پاک کردن تاریخچهٔ جزئیات",
    chart_sample: "فضای نمونه (داده و خط برازش)",
    hint_sample: "نقاط آبی: نمونه‌ها؛ خط مرجانی: مدل فعلی؛ خط چین: مدل تولیدکنندهٔ داده.",
    chart_param: "فضای پارامتر و منظرهٔ تابع هزینه",
    hint_param:
      "رنگ‌ها: هزینهٔ کمتر روشن‌تر؛ ستاره: (w, b)؛ مسیر سبز: GD؛ لوزی: جواب بستهٔ MSE. روی کانتور کلیک کنید.",
    footer: "تهیه توسط Mohammad Khalooei",
    gd_running: "در حال اجرای گرادیان کاهشی…",
    gd_done: "پایان انیمیشن.",
    gd_stopped: "انیمیشن توسط کاربر متوقف شد.",
    theme_dark: "شب",
    theme_light: "روز",
    sidebar_title: "جزئیات محاسبات",
    sidebar_lead:
      "با «ثبت وضعیت فعلی» یا در حین انیمیشن GD، مرحله‌به‌مرحله پارامترها، گرادیان و فرمول‌ها (لاتک) نمایش داده می‌شوند.",
    fab_details: "جزئیات",
    opt_mse_only: "فقط برای تابع MSE معنای حداقل مربعات کلاسیک دارد؛ برای سایر توابع صرفاً مرجع هندسی است.",
    detail_step: "مرحله",
    detail_of: "از",
    detail_params: "پارامترها (قبل از به‌روزرسانی این گام)",
    detail_data: "داده و باقیمانده‌ها",
    detail_loss_val: "مقدار تابع هزینه",
    detail_grad: "گرادیان",
    detail_update: "به‌روزرسانی",
    detail_stopped: "توقف در این نقطه",
    loss_opt_mae: "برای MAE/Huber جواب بستهٔ خطی همان نقطهٔ صورتی (حداقل MSE) نیست؛ فقط مرجع است.",
  },
  en: {
    title: "Visualization & Optimization — Sample vs Parameter Space",
    subtitle: "Linear regression, loss landscape, gradient descent on the loss surface",
    intro_title: "Teaching idea",
    intro_p1:
      "For y ≈ w x + b, sample space is the (x, y) plane; parameter space is the (w, b) plane. The chosen loss defines a landscape over (w, b); gradient descent steps along −∇L (Euclidean geometry in parameter space).",
    bullet_sample: "Sample space: axes x, y; data points; fit line and residuals.",
    bullet_param: "Parameter space: w and b; loss contours; current point and GD path.",
    bullet_loss: "Pick the loss from the control menu; the colored field is that loss over (w, b).",
    repro_note: "Reproducible data (seed), explicit GD steps, per-step LaTeX in the sidebar.",
    controls_title: "Controls",
    label_loss_fn: "Loss function",
    label_huber_delta: "Huber threshold δ",
    label_w: "Slope w",
    label_b: "Intercept b",
    loss_label_mse: "Mean squared error (MSE)",
    loss_label_mae: "Mean absolute error (MAE)",
    loss_label_huber: "Huber loss (mean)",
    gd_title: "Gradient descent (reproducible)",
    gd_steps_desc: "Step: (w, b) ← (w, b) − η ∇L. You can stop anytime; tune the delay between steps.",
    label_lr: "Learning rate η",
    label_steps: "Number of steps",
    label_delay: "Animation delay (ms)",
    label_seed: "Fixed data (seed=42)",
    btn_animate: "Run GD animation",
    btn_stop: "Stop",
    btn_reset: "Reset line",
    btn_opt: "Jump to least squares (MSE)",
    btn_log_state: "Log current state",
    btn_clear_log: "Clear detail log",
    chart_sample: "Sample space (data & fit)",
    hint_sample: "Blue: samples; coral: current fit; dashed: generative reference.",
    chart_param: "Parameter space & loss landscape",
    hint_param: "Colors: lower loss brighter; star: (w,b); green: GD; diamond: closed-form MSE fit. Click contour.",
    footer: "Prepared by Mohammad Khalooei",
    gd_running: "Running gradient descent…",
    gd_done: "Animation finished.",
    gd_stopped: "Animation stopped by user.",
    theme_dark: "Night",
    theme_light: "Day",
    sidebar_title: "Computation details",
    sidebar_lead: "Use “Log current state” or run GD to append step-by-step parameters, gradients, and LaTeX formulas.",
    fab_details: "Details",
    opt_mse_only: "Classical least squares refers to MSE; for other losses the diamond is only a geometric reference.",
    detail_step: "Step",
    detail_of: "of",
    detail_params: "Parameters (before this update)",
    detail_data: "Data & residuals",
    detail_loss_val: "Loss value",
    detail_grad: "Gradient",
    detail_update: "Update",
    detail_stopped: "Stopped here",
    loss_opt_mae: "For MAE/Huber the pink point is the MSE closed form, not the minimizer of the current loss.",
  },
};

let lang = "fa";
let currentLossId = "mse";
let gridCache = null;

function getHuberDelta() {
  const v = parseFloat(document.getElementById("input-huber-delta")?.value || "0.6");
  return Math.max(0.05, Math.min(5, v || 0.6));
}

function lossValue(xs, ys, w, b) {
  if (currentLossId === "mse") return mse(xs, ys, w, b);
  if (currentLossId === "mae") return mae(xs, ys, w, b);
  return huberLoss(xs, ys, w, b, getHuberDelta());
}

function gradLoss(xs, ys, w, b) {
  if (currentLossId === "mse") return gradMse(xs, ys, w, b);
  if (currentLossId === "mae") return gradMae(xs, ys, w, b);
  return gradHuber(xs, ys, w, b, getHuberDelta());
}

function lossGrid(xs, ys, wMin, wMax, bMin, bMax, gw, gb) {
  const Z = [];
  const ws = [];
  const bs = [];
  const hd = getHuberDelta();
  for (let i = 0; i < gw; i++) {
    const row = [];
    const w = wMin + ((wMax - wMin) * i) / (gw - 1);
    ws.push(w);
    for (let j = 0; j < gb; j++) {
      const b = bMin + ((bMax - bMin) * j) / (gb - 1);
      if (i === 0) bs.push(b);
      let val;
      if (currentLossId === "mse") val = mse(xs, ys, w, b);
      else if (currentLossId === "mae") val = mae(xs, ys, w, b);
      else val = huberLoss(xs, ys, w, b, hd);
      row.push(val);
    }
    Z.push(row);
  }
  return { ws, bs, Z };
}

function ensureGrid() {
  if (!gridCache) {
    gridCache = lossGrid(data.xs, data.ys, wGrid.wMin, wGrid.wMax, wGrid.bMin, wGrid.bMax, wGrid.gw, wGrid.gb);
  }
  return gridCache;
}

function invalidateGrid() {
  gridCache = null;
}

function applyI18n() {
  const dict = I18N[lang];
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "fa" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (dict[key] !== undefined) el.textContent = dict[key];
  });
  document.getElementById("lang-fa").classList.toggle("active", lang === "fa");
  document.getElementById("lang-en").classList.toggle("active", lang === "en");
  document.getElementById("lang-fa").setAttribute("aria-pressed", lang === "fa");
  document.getElementById("lang-en").setAttribute("aria-pressed", lang === "en");
  updateLossMetricLabel();
}

function getChartTheme() {
  const light = document.documentElement.getAttribute("data-theme") === "light";
  const fg = light ? "#0f172a" : "#e8edf4";
  const g = light ? "rgba(15,23,42,0.08)" : "rgba(255,255,255,0.06)";
  const z = light ? "rgba(15,23,42,0.2)" : "rgba(255,255,255,0.15)";
  const lc = light ? "rgba(15,23,42,0.15)" : "rgba(255,255,255,0.12)";
  const pbg = light ? "rgba(248,250,252,0.95)" : "rgba(22,29,39,0.6)";
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: pbg,
    font: { color: fg, family: "DM Sans, Vazirmatn, sans-serif", size: 12 },
    xaxis: { gridcolor: g, zerolinecolor: z, linecolor: lc },
    yaxis: { gridcolor: g, zerolinecolor: z, linecolor: lc },
  };
}

function lossColorbarTitle() {
  if (currentLossId === "mse") return "MSE";
  if (currentLossId === "mae") return "MAE";
  return "Huber";
}

function updateLossMetricLabel() {
  const el = document.getElementById("loss-label-display");
  if (!el) return;
  const k = currentLossId === "mse" ? "loss_label_mse" : currentLossId === "mae" ? "loss_label_mae" : "loss_label_huber";
  el.textContent = I18N[lang][k];
}

const W_TRUE = 2;
const B_TRUE = 1;
const N = 28;
const NOISE = 0.35;

let data = generateData(N, W_TRUE, B_TRUE, NOISE, RNG_SEED);
const wGrid = { wMin: -3, wMax: 6, bMin: -2, bMax: 5, gw: 55, gb: 55 };

let wCurrent = 1;
let bCurrent = 0.5;
let trajW = [];
let trajB = [];
let animating = false;
let stopAnimationFlag = false;

const elW = document.getElementById("slider-w");
const elB = document.getElementById("slider-b");
const outW = document.getElementById("out-w");
const outB = document.getElementById("out-b");
const lossVal = document.getElementById("loss-value");
const gdStatus = document.getElementById("gd-status");
const btnAnimate = document.getElementById("btn-animate");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");
const btnOpt = document.getElementById("btn-jump-opt");
const chkSeed = document.getElementById("chk-seed");
const selectLoss = document.getElementById("select-loss");
const huberWrap = document.getElementById("huber-delta-wrap");
const detailsLog = document.getElementById("details-log");
const fabDetails = document.getElementById("fab-details");
const sidebar = document.getElementById("sidebar");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const inputLr = document.getElementById("input-lr");
const inputSteps = document.getElementById("input-steps");
const inputDelay = document.getElementById("input-delay");
const inputHuberDelta = document.getElementById("input-huber-delta");
const btnLogCurrent = document.getElementById("btn-log-current");
const btnClearLog = document.getElementById("btn-clear-log");

function lineSamples(w, b, x0, x1, steps) {
  const xx = [];
  const yy = [];
  for (let i = 0; i <= steps; i++) {
    const t = x0 + ((x1 - x0) * i) / steps;
    xx.push(t);
    yy.push(w * t + b);
  }
  return { xx, yy };
}

function buildSamplePlot() {
  const chartTheme = getChartTheme();
  const { xs, ys } = data;
  const xMin = -0.15,
    xMax = 2.15;
  const line = lineSamples(wCurrent, bCurrent, xMin, xMax, 80);
  const trueLine = lineSamples(W_TRUE, B_TRUE, xMin, xMax, 80);

  const residualX = [];
  const residualY = [];
  for (let i = 0; i < xs.length; i++) {
    residualX.push(xs[i], xs[i], null);
    residualY.push(ys[i], wCurrent * xs[i] + bCurrent, null);
  }

  const traces = [
    {
      type: "scatter",
      mode: "markers",
      x: xs,
      y: ys,
      name: lang === "fa" ? "نمونه‌ها" : "Samples",
      marker: { size: 11, color: "#38bdf8", line: { width: 1, color: "rgba(255,255,255,0.35)" } },
    },
    {
      type: "scatter",
      mode: "lines",
      x: line.xx,
      y: line.yy,
      name: lang === "fa" ? "مدل فعلی" : "Current fit",
      line: { color: "#fb7185", width: 3.2, shape: "linear" },
    },
    {
      type: "scatter",
      mode: "lines",
      x: trueLine.xx,
      y: trueLine.yy,
      name: lang === "fa" ? "مدل مرجع" : "Reference",
      line: { color: "rgba(244,114,182,0.55)", width: 2, dash: "dash" },
    },
    {
      type: "scatter",
      mode: "lines",
      x: residualX,
      y: residualY,
      name: lang === "fa" ? "باقیمانده" : "Residuals",
      line: { color: "rgba(251,191,36,0.45)", width: 1.2 },
      showlegend: true,
    },
  ];

  const yPad = 0.6;
  const yMin = Math.min(...ys, ...line.yy) - yPad;
  const yMax = Math.max(...ys, ...line.yy) + yPad;

  return {
    data: traces,
    layout: {
      ...chartTheme,
      margin: { l: 48, r: 18, t: 28, b: 44 },
      showlegend: true,
      legend: { orientation: "h", y: 1.08, x: 0, font: { size: 11 } },
      xaxis: { ...chartTheme.xaxis, title: { text: "x" }, range: [xMin, xMax] },
      yaxis: { ...chartTheme.yaxis, title: { text: "y" }, range: [yMin, yMax] },
      hovermode: "closest",
    },
    config: { responsive: true, displayModeBar: true, displaylogo: false },
  };
}

function buildParamPlot() {
  const chartTheme = getChartTheme();
  const lightTheme = document.documentElement.getAttribute("data-theme") === "light";
  const g = ensureGrid();
  const traces = [
    {
      type: "contour",
      x: g.bs,
      y: g.ws,
      z: g.Z,
      colorscale: [
        [0, "#0c1220"],
        [0.35, "#1e3a5f"],
        [0.55, "#155e75"],
        [0.72, "#0d9488"],
        [0.88, "#5eead4"],
        [1, "#fef3c7"],
      ],
      contours: { coloring: "fill", showlines: true },
      line: { width: 0.6, color: "rgba(255,255,255,0.12)" },
      name: lossColorbarTitle(),
      showscale: true,
      colorbar: {
        title: lossColorbarTitle(),
        tickfont: { color: lightTheme ? "#475569" : "#cbd5e1", size: 10 },
        titlefont: { color: lightTheme ? "#334155" : "#cbd5e1", size: 11 },
        len: 0.85,
        thickness: 14,
        bgcolor: lightTheme ? "rgba(241,245,249,0.85)" : "rgba(15,20,25,0.35)",
      },
    },
  ];

  if (lightTheme) {
    traces[0].line = { width: 0.5, color: "rgba(15,23,42,0.1)" };
    traces[0].colorscale = [
      [0, "#f8fafc"],
      [0.25, "#e0f2fe"],
      [0.5, "#7dd3fc"],
      [0.75, "#0ea5e9"],
      [1, "#0c4a6e"],
    ];
  }

  if (trajW.length > 1) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: trajB,
      y: trajW,
      name: lang === "fa" ? "مسیر GD" : "GD path",
      line: { color: "#4ade80", width: 2.5 },
    });
  }

  traces.push({
    type: "scatter",
    mode: "markers",
    x: [bCurrent],
    y: [wCurrent],
    name: lang === "fa" ? "(w, b) فعلی" : "Current (w,b)",
    marker: {
      size: 18,
      color: "#fbbf24",
      symbol: "star",
      line: { width: 2, color: "#fff" },
    },
  });

  const ls = closedFormLeastSquares(data.xs, data.ys);
  traces.push({
    type: "scatter",
    mode: "markers",
    x: [ls.b],
    y: [ls.w],
    name: lang === "fa" ? "حداقل MSE (بسته)" : "MSE least squares",
    marker: { size: 10, color: "#f472b6", symbol: "diamond", line: { width: 1, color: "#fff" } },
  });

  return {
    data: traces,
    layout: {
      ...chartTheme,
      margin: { l: 52, r: 24, t: 28, b: 48 },
      showlegend: true,
      legend: { orientation: "h", y: 1.1, x: 0, font: { size: 10 } },
      xaxis: {
        ...chartTheme.xaxis,
        title: { text: "b (intercept)" },
        range: [wGrid.bMin, wGrid.bMax],
        scaleanchor: "y",
        scaleratio: 1,
      },
      yaxis: {
        ...chartTheme.yaxis,
        title: { text: "w (slope)" },
        range: [wGrid.wMin, wGrid.wMax],
      },
      hovermode: "closest",
    },
    config: { responsive: true, displayModeBar: true, displaylogo: false },
  };
}

function render() {
  const L = lossValue(data.xs, data.ys, wCurrent, bCurrent);
  lossVal.textContent = L.toFixed(5);
  outW.textContent = wCurrent.toFixed(2);
  outB.textContent = bCurrent.toFixed(2);

  const s1 = buildSamplePlot();
  const s2 = buildParamPlot();
  Plotly.react("plot-sample", s1.data, s1.layout, s1.config);
  Plotly.react("plot-param", s2.data, s2.layout, s2.config);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function setAnimatingControls(on) {
  const dis = !!on;
  elW.disabled = dis;
  elB.disabled = dis;
  selectLoss.disabled = dis;
  inputHuberDelta.disabled = dis;
  inputLr.disabled = dis;
  inputSteps.disabled = dis;
  inputDelay.disabled = dis;
  chkSeed.disabled = dis;
  btnReset.disabled = dis;
  btnOpt.disabled = dis;
}

function latexLossDefinition() {
  const d = getHuberDelta();
  if (currentLossId === "mse") {
    return String.raw`\[ \mathcal{L}_{\mathrm{MSE}}(w,b) = \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - w x_i - b\bigr)^2 \]`;
  }
  if (currentLossId === "mae") {
    return String.raw`\[ \mathcal{L}_{\mathrm{MAE}}(w,b) = \frac{1}{n}\sum_{i=1}^{n}\bigl|y_i - w x_i - b\bigr| \]`;
  }
  return (
    String.raw`\[ \mathcal{L}_{\delta}(w,b) = \frac{1}{n}\sum_{i=1}^{n} \ell_\delta(r_i),\quad r_i=y_i-wx_i-b,\quad \ell_\delta(r)=\begin{cases}\tfrac12 r^2 & |r|\le\delta \\ \delta(|r|-\tfrac12\delta) & |r|>\delta \end{cases} \]` +
    `<p class="small muted" style="margin-top:0.35rem;text-align:center">$\\delta = ${d.toFixed(4)}$</p>`
  );
}

function latexGradDefinition() {
  if (currentLossId === "mse") {
    return String.raw`\[ \frac{\partial \mathcal{L}}{\partial w} = -\frac{2}{n}\sum_i x_i r_i,\quad \frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n}\sum_i r_i,\quad r_i=y_i-wx_i-b \]`;
  }
  if (currentLossId === "mae") {
    return String.raw`\[ g_w=\frac{\partial \mathcal{L}}{\partial w}= -\frac{1}{n}\sum_i x_i\,\mathrm{sign}(r_i),\quad g_b= -\frac{1}{n}\sum_i \mathrm{sign}(r_i) \]`;
  }
  return String.raw`\[ g_w= -\frac{1}{n}\sum_i x_i\,\psi_{\delta}(r_i),\quad g_b= -\frac{1}{n}\sum_i \psi_{\delta}(r_i),\quad \psi_{\delta}(r)=\begin{cases} r & |r|\le\delta \\ \delta\,\mathrm{sign}(r) & |r|>\delta \end{cases} \]`;
}

function latexUpdateRule() {
  return String.raw`\[ w \leftarrow w - \eta\,g_w,\qquad b \leftarrow b - \eta\,g_b \]`;
}

async function waitMathJaxReady() {
  if (!window.MathJax) return;
  if (window.MathJax.startup && window.MathJax.startup.promise) {
    try {
      await window.MathJax.startup.promise;
    } catch (_) {}
  }
}

async function typesetElement(el) {
  await waitMathJaxReady();
  if (window.MathJax && window.MathJax.typesetPromise) {
    try {
      await window.MathJax.typesetPromise([el]);
    } catch (e) {
      console.warn("MathJax typeset", e);
    }
  }
}

function buildDetailBlock(opts) {
  const {
    stepIndex,
    totalSteps,
    wBefore,
    bBefore,
    wAfter,
    bAfter,
    lr,
    gw,
    gb,
    Lbefore,
    tag,
  } = opts;
  const st = residualStats(data.xs, data.ys, wBefore, bBefore);
  const isFa = lang === "fa";

  const title =
    tag === "manual"
      ? isFa
        ? "ثبت دستی وضعیت"
        : "Manual log"
      : `${I18N[lang].detail_step} ${stepIndex} ${I18N[lang].detail_of} ${totalSteps}`;

  const lossDef = latexLossDefinition();
  const gradDef = latexGradDefinition();
  const updRule = latexUpdateRule();

  const samplePreview = st.rs
    .slice(0, 4)
    .map((r, i) => `r_{${i + 1}}=${r.toFixed(3)}`)
    .join(",\\; ");

  const wrap = document.createElement("div");
  wrap.className = "detail-step";
  wrap.innerHTML = `
    <h3>${escapeHtml(title)}</h3>
    <dl class="kv">
      <dt>${escapeHtml(I18N[lang].detail_params)}</dt>
      <dd>$w=${wBefore.toFixed(4)},\\; b=${bBefore.toFixed(4)}$</dd>
      <dt>${escapeHtml(I18N[lang].detail_data)}</dt>
      <dd>$n=${st.n},\\; x\\in[${st.minX.toFixed(2)},\\,${st.maxX.toFixed(2)}],\\; \\overline{|r|}=${st.meanAbs.toFixed(4)},\\; \\overline{r^2}=${st.meanSq.toFixed(4)}$</dd>
      <dt>${escapeHtml(I18N[lang].detail_loss_val)}</dt>
      <dd>$\\mathcal{L}=${Lbefore.toFixed(6)}$</dd>
      <dt>${escapeHtml(I18N[lang].detail_grad)}</dt>
      <dd>$g_w=${gw.toFixed(6)},\\; g_b=${gb.toFixed(6)}$</dd>
      <dt>${escapeHtml(I18N[lang].detail_update)}</dt>
      <dd>$\\eta=${lr.toFixed(4)},\\; w'=${wAfter.toFixed(4)},\\; b'=${bAfter.toFixed(4)}$</dd>
    </dl>
    <div class="mj-wrap">${lossDef}</div>
    <div class="mj-wrap">${gradDef}</div>
    <div class="mj-wrap">${updRule}</div>
    <p class="small muted" style="margin-top:0.5rem">${isFa ? "نمونهٔ باقیمانده‌ها (چند جملهٔ اول):" : "Sample residual labels (first few):"} $${samplePreview}$</p>
    <hr class="step-sep" />
  `;
  return wrap;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function appendDetailFromState(opts) {
  const wrap = buildDetailBlock(opts);
  detailsLog.appendChild(wrap);
  await typesetElement(wrap);
  detailsLog.scrollTop = detailsLog.scrollHeight;
}

async function appendStoppedBlock(completedSteps, totalSteps) {
  const wrap = document.createElement("div");
  wrap.className = "detail-step";
  wrap.innerHTML = `<h3>${escapeHtml(I18N[lang].detail_stopped)}</h3>
    <p class="small">${escapeHtml(I18N[lang].gd_stopped)} (${completedSteps}/${totalSteps})</p>
    <hr class="step-sep" />`;
  detailsLog.appendChild(wrap);
  detailsLog.scrollTop = detailsLog.scrollHeight;
}

function openSidebar() {
  document.body.classList.add("sidebar-open");
  sidebar.setAttribute("aria-hidden", "false");
  if (window.matchMedia("(max-width: 720px)").matches) {
    sidebarBackdrop.hidden = false;
  }
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
  sidebar.setAttribute("aria-hidden", "true");
  sidebarBackdrop.hidden = true;
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  document.getElementById("theme-dark").classList.toggle("active", theme === "dark");
  document.getElementById("theme-light").classList.toggle("active", theme === "light");
  document.getElementById("theme-dark").setAttribute("aria-pressed", theme === "dark");
  document.getElementById("theme-light").setAttribute("aria-pressed", theme === "light");
  try {
    localStorage.setItem("kvo-theme", theme);
  } catch (_) {}
  render();
}

async function runAnimation() {
  if (animating) return;
  animating = true;
  stopAnimationFlag = false;
  btnAnimate.disabled = true;
  setAnimatingControls(true);
  btnStop.style.display = "";
  gdStatus.textContent = I18N[lang].gd_running;

  const lr = parseFloat(document.getElementById("input-lr").value) || 0.15;
  const steps = Math.max(1, parseInt(document.getElementById("input-steps").value, 10) || 40);
  const delay = Math.max(50, parseInt(document.getElementById("input-delay").value, 10) || 1200);

  trajW = [wCurrent];
  trajB = [bCurrent];

  const wLo = parseFloat(elW.min),
    wHi = parseFloat(elW.max);
  const bLo = parseFloat(elB.min),
    bHi = parseFloat(elB.max);

  for (let t = 0; t < steps; t++) {
    if (stopAnimationFlag) {
      await appendStoppedBlock(t, steps);
      break;
    }

    const wBefore = wCurrent;
    const bBefore = bCurrent;
    const Lb = lossValue(data.xs, data.ys, wBefore, bBefore);
    const { gw, gb } = gradLoss(data.xs, data.ys, wBefore, bBefore);

    wCurrent = clamp(wBefore - lr * gw, wLo, wHi);
    bCurrent = clamp(bBefore - lr * gb, bLo, bHi);
    elW.value = String(wCurrent);
    elB.value = String(bCurrent);
    trajW.push(wCurrent);
    trajB.push(bCurrent);
    render();

    await appendDetailFromState({
      stepIndex: t + 1,
      totalSteps: steps,
      wBefore,
      bBefore,
      wAfter: wCurrent,
      bAfter: bCurrent,
      lr,
      gw,
      gb,
      Lbefore: Lb,
      tag: "gd",
    });

    const La = lossValue(data.xs, data.ys, wCurrent, bCurrent);
    gdStatus.textContent =
      (lang === "fa" ? `گام ` : `Step `) +
      (t + 1) +
      "/" +
      steps +
      " — L = " +
      La.toFixed(5);

    await sleep(delay);
  }

  animating = false;
  setAnimatingControls(false);
  btnAnimate.disabled = false;
  btnStop.style.display = "none";
  gdStatus.textContent = stopAnimationFlag ? I18N[lang].gd_stopped : I18N[lang].gd_done;
}

function onLangChange(newLang) {
  lang = newLang;
  applyI18n();
  render();
}

elW.addEventListener("input", () => {
  if (animating) return;
  wCurrent = parseFloat(elW.value);
  trajW = [];
  trajB = [];
  render();
});
elB.addEventListener("input", () => {
  if (animating) return;
  bCurrent = parseFloat(elB.value);
  trajW = [];
  trajB = [];
  render();
});

btnAnimate.addEventListener("click", runAnimation);
btnStop.addEventListener("click", () => {
  stopAnimationFlag = true;
});

btnReset.addEventListener("click", () => {
  if (animating) return;
  wCurrent = 1;
  bCurrent = 0.5;
  elW.value = "1";
  elB.value = "0.5";
  trajW = [];
  trajB = [];
  gdStatus.textContent = "";
  render();
});

btnOpt.addEventListener("click", () => {
  const ls = closedFormLeastSquares(data.xs, data.ys);
  wCurrent = ls.w;
  bCurrent = ls.b;
  elW.value = String(clamp(wCurrent, parseFloat(elW.min), parseFloat(elW.max)));
  elB.value = String(clamp(bCurrent, parseFloat(elB.min), parseFloat(elB.max)));
  trajW = [];
  trajB = [];
  let msg = lang === "fa" ? "پرش به حداقل MSE (بسته)." : "Jumped to closed-form MSE solution.";
  if (currentLossId !== "mse") msg += " " + I18N[lang].loss_opt_mae;
  gdStatus.textContent = msg;
  render();
});

chkSeed.addEventListener("change", () => {
  const seed = chkSeed.checked ? RNG_SEED : Date.now() >>> 0;
  data = generateData(N, W_TRUE, B_TRUE, NOISE, seed);
  invalidateGrid();
  ensureGrid();
  trajW = [];
  trajB = [];
  render();
});

selectLoss.addEventListener("change", () => {
  if (animating) return;
  currentLossId = selectLoss.value;
  invalidateGrid();
  huberWrap.classList.toggle("hidden", currentLossId !== "huber");
  updateLossMetricLabel();
  render();
});

inputHuberDelta.addEventListener("change", () => {
  if (animating) return;
  if (currentLossId === "huber") {
    invalidateGrid();
    render();
  }
});

document.getElementById("lang-fa").addEventListener("click", () => onLangChange("fa"));
document.getElementById("lang-en").addEventListener("click", () => onLangChange("en"));

document.getElementById("theme-dark").addEventListener("click", () => setTheme("dark"));
document.getElementById("theme-light").addEventListener("click", () => setTheme("light"));

fabDetails.addEventListener("click", () => openSidebar());
document.getElementById("btn-close-sidebar").addEventListener("click", () => closeSidebar());
sidebarBackdrop.addEventListener("click", () => closeSidebar());

btnClearLog.addEventListener("click", () => {
  detailsLog.innerHTML = "";
});

btnLogCurrent.addEventListener("click", async () => {
  const { gw, gb } = gradLoss(data.xs, data.ys, wCurrent, bCurrent);
  const lr = parseFloat(document.getElementById("input-lr").value) || 0.15;
  const wN = clamp(wCurrent - lr * gw, parseFloat(elW.min), parseFloat(elW.max));
  const bN = clamp(bCurrent - lr * gb, parseFloat(elB.min), parseFloat(elB.max));
  const Lb = lossValue(data.xs, data.ys, wCurrent, bCurrent);
  await appendDetailFromState({
    stepIndex: 0,
    totalSteps: 0,
    wBefore: wCurrent,
    bBefore: bCurrent,
    wAfter: wN,
    bAfter: bN,
    lr,
    gw,
    gb,
    Lbefore: Lb,
    tag: "manual",
  });
});

window.addEventListener("resize", () => {
  Plotly.Plots.resize(document.getElementById("plot-sample"));
  Plotly.Plots.resize(document.getElementById("plot-param"));
});

try {
  const t = localStorage.getItem("kvo-theme");
  if (t === "light" || t === "dark") setTheme(t);
} catch (_) {}

huberWrap.classList.toggle("hidden", currentLossId !== "huber");
applyI18n();
selectLoss.value = currentLossId;

const s0 = buildSamplePlot();
const p0 = buildParamPlot();
Plotly.newPlot("plot-sample", s0.data, s0.layout, s0.config);
Plotly.newPlot("plot-param", p0.data, p0.layout, p0.config);
render();

document.getElementById("plot-param").on("plotly_click", (ev) => {
  if (animating) return;
  const pt = ev.points && ev.points[0];
  if (!pt || pt.x === undefined || pt.y === undefined) return;
  if (pt.curveNumber !== 0) return;
  bCurrent = clamp(pt.x, parseFloat(elB.min), parseFloat(elB.max));
  wCurrent = clamp(pt.y, parseFloat(elW.min), parseFloat(elW.max));
  elB.value = String(bCurrent);
  elW.value = String(wCurrent);
  trajW = [];
  trajB = [];
  render();
});
