(function () {
  const $ = (sel) => document.querySelector(sel);

  const form = $('#pricing-form');
  const submitBtn = $('#submit-btn');
  const loading = $('#loading');
  const errorBox = $('#error-box');
  const predsSection = $('#predictions');
  const linearValue = $('#linear-value');
  const respTimeEl = $('#response-time');
  const sourceEl = $('#display-source');
  const displaySelect = $('#display-format');
  const year = $('#year');
  if (year) year.textContent = new Date().getFullYear();

  // Config
  const USD_TO_INR_RATE = 83.0; // keep in sync with backend
  const SYNC_INTERVAL_MS = 60000; // periodic server sync
  const DEBOUNCE_MS = 200; // debounce for format changes

  // Runtime state
  let lastFormData = null;
  let lastBaseUSD = null; // canonical numeric in USD from server
  let lastResponseMs = null; // last server response time
  let syncTimer = null;

  function show(el) { el && el.classList.remove('hidden'); }
  function hide(el) { el && el.classList.add('hidden'); }

  function setSkeleton(state) {
    if (!linearValue) return;
    if (state) {
      linearValue.classList.add('skeleton');
      linearValue.textContent = ' ';
    } else {
      linearValue.classList.remove('skeleton');
    }
  }

  function setLoading(state) {
    if (state) {
      if (submitBtn) submitBtn.textContent = 'Predicting…';
      submitBtn.disabled = true;
      show(loading);
      hide(errorBox);
      setSkeleton(true);
    } else {
      if (submitBtn) submitBtn.textContent = 'Predict Price';
      submitBtn.disabled = false;
      hide(loading);
      setSkeleton(false);
    }
  }

  function setError(message) {
    errorBox.textContent = message || 'Something went wrong.';
    show(errorBox);
  }

  function getFormDataObject() {
    const data = Object.fromEntries(new FormData(form).entries());
    if (typeof data.current_price !== 'undefined') data.current_price = Number(data.current_price);
    if (typeof data.competitor_price !== 'undefined') data.competitor_price = Number(data.competitor_price);
    if (typeof data.stock !== 'undefined') data.stock = Number(data.stock);
    data.display = displaySelect ? displaySelect.value : 'usd';
    return data;
  }

  async function requestPrediction(formData) {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });
    const json = await res.json();
    if (!res.ok || !json.success) {
      const msg = (json && (json.message || json.errors && json.errors.join(', '))) || 'Prediction failed';
      throw new Error(msg);
    }
    return json;
  }

  function formatUSD(n) {
    return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function formatINRGrouping(n) {
    const s = Math.abs(Number(n)).toFixed(2);
    const [whole, frac] = s.split('.');
    let g = whole;
    if (whole.length > 3) {
      const last3 = whole.slice(-3);
      let rest = whole.slice(0, -3);
      const parts = [];
      while (rest.length > 2) {
        parts.unshift(rest.slice(-2));
        rest = rest.slice(0, -2);
      }
      if (rest) parts.unshift(rest);
      g = parts.join(',') + ',' + last3;
    }
    const sign = Number(n) < 0 ? '-' : '';
    return sign + '₹' + g + '.' + frac;
  }

  function numberToWordsIntl(num) {
    const a = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
    const b = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
    function inWords(n) {
      if (n < 20) return a[n];
      if (n < 100) return b[Math.floor(n / 10)] + (n % 10 ? '-' + a[n % 10] : '');
      if (n < 1000) return a[Math.floor(n / 100)] + ' hundred' + (n % 100 ? ' ' + inWords(n % 100) : '');
      if (n < 1e6) return inWords(Math.floor(n / 1e3)) + ' thousand' + (n % 1e3 ? ' ' + inWords(n % 1e3) : '');
      if (n < 1e9) return inWords(Math.floor(n / 1e6)) + ' million' + (n % 1e6 ? ' ' + inWords(n % 1e6) : '');
      if (n < 1e12) return inWords(Math.floor(n / 1e9)) + ' billion' + (n % 1e9 ? ' ' + inWords(n % 1e9) : '');
      return String(n);
    }
    const whole = Math.floor(Math.abs(num));
    const frac = Math.round((Math.abs(num) - whole) * 100);
    const words = inWords(whole) || 'zero';
    const sign = num < 0 ? 'negative ' : '';
    return sign + words + (frac ? ` and ${String(frac).padStart(2, '0')}/100` : '');
  }

  function numberToWordsIndian(num) {
    const a = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
    const b = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
    function two(n) { return n < 20 ? a[n] : b[Math.floor(n/10)] + (n % 10 ? '-' + a[n % 10] : ''); }
    function three(n) { return n < 100 ? two(n) : a[Math.floor(n/100)] + ' hundred' + (n % 100 ? ' ' + two(n % 100) : ''); }
    const whole = Math.floor(Math.abs(num));
    const frac = Math.round((Math.abs(num) - whole) * 100);
    const crore = Math.floor(whole / 10000000);
    const lakh = Math.floor((whole % 10000000) / 100000);
    const thousand = Math.floor((whole % 100000) / 1000);
    const hundred = Math.floor(whole % 1000);
    const parts = [];
    if (crore) parts.push(three(crore) + ' crore');
    if (lakh) parts.push(three(lakh) + ' lakh');
    if (thousand) parts.push(three(thousand) + ' thousand');
    if (hundred) parts.push(three(hundred));
    const words = parts.length ? parts.join(' ') : 'zero';
    const sign = num < 0 ? 'negative ' : '';
    return sign + words + (frac ? ` and ${String(frac).padStart(2, '0')}/100` : '');
  }

  function updateResponseTime(ms) {
    if (respTimeEl) respTimeEl.textContent = (typeof ms === 'number') ? Math.round(ms) : '—';
  }

  function applyLocalDisplay() {
    if (lastBaseUSD == null) return;
    const mode = displaySelect ? displaySelect.value : 'usd';
    if (mode === 'usd') {
      linearValue.textContent = formatUSD(lastBaseUSD);
    } else if (mode === 'inr') {
      const inr = Number(lastBaseUSD) * USD_TO_INR_RATE;
      linearValue.textContent = formatINRGrouping(inr);
    } else if (mode === 'words-intl') {
      linearValue.textContent = numberToWordsIntl(lastBaseUSD);
    } else if (mode === 'words-indian') {
      const inr = Number(lastBaseUSD) * USD_TO_INR_RATE;
      linearValue.textContent = numberToWordsIndian(inr);
    }
    // Force cached indicator and reset response time
    if (sourceEl) sourceEl.textContent = 'cached';
    if (respTimeEl) respTimeEl.textContent = '—';
    show(predsSection);
  }

  function debounce(fn, wait) {
    let t; return function(...args) { clearTimeout(t); t = setTimeout(() => fn.apply(this, args), wait); };
  }

  async function requestAndRender(opts = { silent: false }) {
    const { silent } = opts || {};
    if (!silent) setLoading(true);
    try {
      const body = lastFormData || getFormDataObject();
      body.display = displaySelect ? displaySelect.value : 'usd';
      const json = await requestPrediction(body);
      lastBaseUSD = json && json.predictions ? Number(json.predictions.linear_regression) : null;
      lastResponseMs = (typeof json.duration_ms === 'number') ? json.duration_ms : lastResponseMs;
      const display = json.display;
      linearValue.textContent = display && display.value ? display.value : '—';
      updateResponseTime(lastResponseMs);
      if (sourceEl) sourceEl.textContent = 'server';
      show(predsSection);
    } catch (err) {
      console.error(err);
      if (!silent) {
        setError(err.message || 'Network error while predicting.');
        hide(predsSection);
      }
    } finally {
      if (!silent) setLoading(false);
    }
  }

  // Submit triggers server fetch and primes cache
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    lastFormData = getFormDataObject();
    await requestAndRender({ silent: false });
  });

  // Debounced local display switch, plus background server refresh
  if (displaySelect) {
    const onChange = () => {
      if (!lastFormData) return; // no-op until first prediction
      // Immediate local cached update
      applyLocalDisplay();
      // Background refresh shortly after to get server-formatted value and response time
      setTimeout(() => {
        requestAndRender({ silent: true });
      }, 150);
    };
    displaySelect.addEventListener('change', onChange);
  }

  // Periodic server sync to keep cache fresh
  function startSync() {
    if (syncTimer) clearInterval(syncTimer);
    syncTimer = setInterval(() => {
      if (!lastFormData) return;
      requestAndRender({ silent: true });
    }, SYNC_INTERVAL_MS);
  }
  startSync();
})();