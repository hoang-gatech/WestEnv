'use strict';

// Everything that comes back from the API or from OpenStreetMap (place names, river names) is
// untrusted text, so the page is built with textContent via h(), never with innerHTML.

const TILES = {
  osm: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  dem: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  soil:
    'https://maps.isric.org/mapserv?map=/map/wrb.map&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap' +
    '&LAYERS=MostProbable&STYLES=&CRS=EPSG:3857&BBOX={bbox-epsg-3857}&WIDTH=256&HEIGHT=256' +
    '&FORMAT=image/png&TRANSPARENT=true',
};
const SOIL_LEGEND =
  'https://maps.isric.org/mapserv?map=/map/wrb.map&version=1.3.0&service=WMS&request=GetLegendGraphic' +
  '&sld_version=1.1.0&layer=MostProbable&format=image/png&STYLE=default';

// Sections fetched automatically on click. Water is on demand: the Overpass API behind it
// took ~35 s per call when measured, so it never runs unless asked.
const AUTO_SECTIONS = ['place', 'weather', 'terrain', 'soil', 'hotspots'];
const TITLES = {
  place: 'Place',
  weather: 'Weather',
  terrain: 'Terrain',
  soil: 'Soil',
  hotspots: 'Known dioxin hotspots',
  water: 'Nearby surface water',
};

const $ = (selector) => document.querySelector(selector);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) el.setAttribute(key, value);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
  return el;
}

const num = (v, digits = 1) => (v == null || Number.isNaN(Number(v)) ? '–' : Number(v).toFixed(digits));
const spinner = (text) => h('span', {}, h('span', { class: 'spinner' }), text);

function kv(rows) {
  return h('dl', { class: 'kv' }, rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));
}

function table(head, rows) {
  return h(
    'table',
    {},
    h('thead', {}, h('tr', {}, head.map((c) => h('th', {}, c)))),
    h('tbody', {}, rows.map((r) => h('tr', {}, r.map((c) => h('td', {}, c))))),
  );
}

function provenance(d) {
  const link = (s) =>
    /^https?:\/\//.test(s.url) ? h('a', { href: s.url, target: '_blank', rel: 'noopener' }, s.name) : s.name;
  return h(
    'details',
    {},
    h('summary', {}, `Sources & caveats (${d.caveats.length})`),
    h('ul', {}, d.caveats.map((c) => h('li', {}, c))),
    h('ul', {}, d.sources.map((s) => h('li', {}, link(s), ` · retrieved ${new Date(s.retrieved_at).toLocaleString()}`))),
  );
}

const RENDER = {
  place: (d) => [
    h('div', { class: 'big' }, d.province || 'No address here'),
    d.locality && h('div', {}, d.locality),
    d.display_name && h('div', { class: 'muted small' }, d.display_name),
  ],

  weather: (d) => [
    h('div', { class: 'big' }, `${num(d.current.temperature_c)}°C`, h('span', { class: 'muted' }, ` ${d.current.condition}`)),
    kv([
      ['Humidity', `${num(d.current.humidity_pct, 0)}%`],
      ['Rain now', `${num(d.current.precipitation_mm)} mm`],
      ['Wind', `${num(d.current.wind_speed_kmh, 0)} km/h`],
    ]),
    table(
      ['Day', 'Min', 'Max', 'Rain', 'P(rain)'],
      d.daily.map((x) => [
        x.date.slice(5),
        `${num(x.temp_min_c, 0)}°`,
        `${num(x.temp_max_c, 0)}°`,
        `${num(x.precipitation_mm)} mm`,
        x.precipitation_probability_pct == null ? '–' : `${x.precipitation_probability_pct}%`,
      ]),
    ),
  ],

  terrain: (d) => [
    kv([
      ['Elevation', `${num(d.elevation_m, 0)} m`],
      [
        'Slope',
        d.downslope_direction
          ? `${num(d.slope_percent)}% — water runs ${d.downslope_direction} (${d.downslope_bearing_deg}°)`
          : `${num(d.slope_percent)}% — flat at DEM resolution`,
      ],
    ]),
  ],

  soil(d) {
    const out = [];
    if (d.most_probable_wrb_class) {
      const [top, ...rest] = d.wrb_class_probabilities;
      out.push(h('div', { class: 'big' }, d.most_probable_wrb_class, h('span', { class: 'muted' }, ` ${top ? top.probability_pct + '%' : ''}`)));
      if (rest.length) out.push(h('div', { class: 'muted small' }, 'Also possible: ' + rest.map((p) => `${p.name} ${p.probability_pct}%`).join(', ')));
    }
    const low = d.caveats.find((c) => c.startsWith('Low confidence'));
    if (low) out.push(h('div', { class: 'flag' }, low));
    if (d.depths.length) {
      out.push(
        table(
          ['Depth', 'Texture', 'Clay %', 'Silt %', 'Sand %', 'pH', 'OC g/kg'],
          d.depths.map((x) => [x.depth, x.usda_texture_class || '–', num(x.clay_pct, 0), num(x.silt_pct, 0), num(x.sand_pct, 0), num(x.ph_h2o), num(x.organic_carbon_g_per_kg)]),
        ),
      );
    } else {
      out.push(h('p', { class: 'muted' }, 'No soil values at this point.'));
    }
    return out;
  },

  hotspots: (d) => [
    h('div', { class: 'big' }, `${num(d.nearest.distance_km)} km ${d.nearest.direction}`, h('span', { class: 'muted' }, ` to ${d.nearest.name}`)),
    h('div', { class: 'muted small' }, d.nearest.province),
    d.within_radius.length === 0 && h('div', { class: 'muted small' }, `None within ${d.radius_km} km.`),
  ],

  water(d) {
    if (!d.features.length) {
      return [h('p', { class: 'muted' }, `No mapped water within ${d.radius_m} m. OpenStreetMap is incomplete in rural areas, so this does not mean there is none.`)];
    }
    return [
      h('div', { class: 'big' }, `${d.nearest.distance_m} m`, h('span', { class: 'muted' }, ` ${d.nearest.direction || ''} · ${d.nearest.name || d.nearest.kind}`)),
      table(['Type', 'Name', 'Dist.', 'Dir.'], d.features.map((f) => [f.kind, f.name || '–', `${f.distance_m} m`, f.direction || '–'])),
    ];
  },
};

// --- map -------------------------------------------------------------------------------------

if (typeof maplibregl === 'undefined') {
  $('#map').textContent = 'The map library could not be loaded (needs access to unpkg.com).';
  throw new Error('maplibregl is not available');
}

const demSource = { type: 'raster-dem', tiles: [TILES.dem], tileSize: 256, maxzoom: 14, encoding: 'terrarium' };
const map = new maplibregl.Map({
  container: 'map',
  center: [106.0, 16.3],
  zoom: 5,
  minZoom: 4,
  maxZoom: 17,
  maxPitch: 70,
  maxBounds: [[96, 3], [118, 27]],
  style: {
    version: 8,
    sources: {
      osm: {
        type: 'raster',
        tiles: [TILES.osm],
        tileSize: 256,
        maxzoom: 19,
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      },
      soil: {
        type: 'raster',
        tiles: [TILES.soil],
        tileSize: 256,
        attribution: 'Soil: <a href="https://soilgrids.org">ISRIC SoilGrids</a> (CC BY 4.0)',
      },
      // two DEM sources: one drives the 3D terrain, one the hillshade (MapLibre's recommendation)
      dem: { ...demSource, attribution: 'Terrain: AWS Terrain Tiles (Copernicus, SRTM and others)' },
      'dem-shade': demSource,
    },
    layers: [
      { id: 'osm', type: 'raster', source: 'osm' },
      { id: 'soil', type: 'raster', source: 'soil', layout: { visibility: 'none' }, paint: { 'raster-opacity': 0.65 } },
      { id: 'hillshade', type: 'hillshade', source: 'dem-shade', layout: { visibility: 'none' }, paint: { 'hillshade-exaggeration': 0.55 } },
    ],
  },
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'bottom-left');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

const pin = new maplibregl.Marker({ element: h('div', { class: 'pin' }) });
let token = 0; // bumped on every click so a slow response for an old point is ignored
let controller = null;

const panelBody = $('#panel-body');

function note(text) {
  const el = $('#layers-note');
  el.textContent = text || '';
  el.hidden = !text;
  if (text) $('#layers').open = true; // a failure message inside a collapsed card would go unseen
}

function header(lat, lon) {
  const text = `${lat}, ${lon}`;
  const copy = h('button', {
    type: 'button',
    onclick: async () => {
      try {
        await navigator.clipboard.writeText(text);
        copy.textContent = 'Copied';
      } catch {
        copy.textContent = 'Copy failed';
      }
      setTimeout(() => (copy.textContent = 'Copy'), 1500);
    },
  }, 'Copy');
  const input = h('input', { type: 'text', placeholder: 'Go to: lat, lon', 'aria-label': 'Go to coordinates' });
  const form = h('form', { id: 'goto', onsubmit: (ev) => {
    ev.preventDefault();
    const m = input.value.match(/^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$/);
    if (m && Math.abs(+m[1]) <= 90 && Math.abs(+m[2]) <= 180) selectPoint(+m[1], +m[2], true);
    else input.setCustomValidity('Enter "lat, lon" in decimal degrees');
  } }, input, h('button', { type: 'submit' }, 'Go'));
  input.addEventListener('input', () => input.setCustomValidity(''));
  return h('div', {}, h('h2', {}, 'Site profile'), h('div', { class: 'coords' }, h('code', {}, text), copy), form);
}

function skeleton(name) {
  return h('section', { class: 'section', id: `sec-${name}` }, h('h3', {}, TITLES[name]), h('div', { class: 'body' }, spinner('Loading…')));
}

function waterSkeleton(lat, lon, myToken) {
  const body = h('div', { class: 'body' });
  const button = h('button', { type: 'button', onclick: () => {
    body.replaceChildren(spinner('Querying OpenStreetMap, can take up to a minute…'));
    load('water', lat, lon, myToken, controller.signal);
  } }, 'Find nearby water');
  body.append(button, h('span', { class: 'muted small' }, '  slow: public Overpass API'));
  return h('section', { class: 'section', id: 'sec-water' }, h('h3', {}, TITLES.water), body);
}

async function load(name, lat, lon, myToken, signal) {
  const body = $(`#sec-${name} .body`);
  try {
    const res = await fetch(`/api/point/${name}?lat=${lat}&lon=${lon}`, { signal });
    const data = await res.json().catch(() => ({}));
    if (myToken !== token) return;
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    body.replaceChildren(...[RENDER[name](data)].flat().filter(Boolean), provenance(data));
  } catch (err) {
    if (err.name === 'AbortError' || myToken !== token) return;
    body.replaceChildren(h('div', { class: 'error' }, `Unavailable: ${err.message}`));
  }
}

function selectPoint(lat, lon, fly = false) {
  lat = Number(lat.toFixed(5));
  lon = Number(lon.toFixed(5));
  token += 1;
  const myToken = token;
  if (controller) controller.abort();
  controller = new AbortController();
  history.replaceState(null, '', `#${lat},${lon}`);
  pin.setLngLat([lon, lat]).addTo(map);
  if (fly) map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 9) });
  $('#panel-empty').hidden = true;
  panelBody.hidden = false;
  panelBody.replaceChildren(header(lat, lon), ...AUTO_SECTIONS.map(skeleton), waterSkeleton(lat, lon, myToken));
  $('#panel').scrollTop = 0;
  for (const name of AUTO_SECTIONS) load(name, lat, lon, myToken, controller.signal);
}

// --- layers ----------------------------------------------------------------------------------

// Layer calls need the style, not the tiles. MapLibre's own 'load' event waits for the first
// tiles and is not re-fired if they all fail (a blocked or offline base-tile server), so nothing
// here may depend on it: the click handler and controls are wired straight away.
let styleReady = false;
map.once('style.load', () => {
  styleReady = true;
});
const whenStyleReady = (fn) => (styleReady ? fn() : map.once('style.load', fn));

function setBase(mode) {
  const terrain = mode === 'terrain';
  whenStyleReady(() => {
    map.setLayoutProperty('hillshade', 'visibility', terrain ? 'visible' : 'none');
    map.setTerrain(terrain ? { source: 'dem', exaggeration: 1.4 } : null);
    map.easeTo({ pitch: terrain ? 55 : 0, duration: 800 });
  });
}

function setSoil(on) {
  whenStyleReady(() => map.setLayoutProperty('soil', 'visibility', on ? 'visible' : 'none'));
  const legend = $('#soil-legend');
  if (on && !legend.src) legend.src = SOIL_LEGEND;
  legend.hidden = !on;
}

const tempClass = (t) => (t < 15 ? 'wx-cold' : t < 25 ? 'wx-mild' : t < 32 ? 'wx-warm' : 'wx-hot');
let weatherMarkers = null;
let hotspotMarkers = null;

function labelMarker(marker, label) {
  marker.getElement().setAttribute('aria-label', label);
}

async function getGeoJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(((await res.json().catch(() => ({}))).detail) || `HTTP ${res.status}`);
  return res.json();
}

async function setWeather(on) {
  if (on && !weatherMarkers) {
    try {
      const geojson = await getGeoJSON('/api/overview/weather');
      weatherMarkers = geojson.features.map((f) => {
        const p = f.properties;
        const label = `${p.name}: ${num(p.temperature_c)}°C, ${p.condition}, rain ${num(p.precipitation_mm)} mm, wind ${num(p.wind_speed_kmh, 0)} km/h`;
        const el = h('div', { class: `wx ${tempClass(p.temperature_c)}${p.precipitation_mm > 0.1 ? ' wx-rain' : ''}`, title: label }, `${Math.round(p.temperature_c)}°`);
        const marker = new maplibregl.Marker({ element: el }).setLngLat(f.geometry.coordinates).addTo(map);
        labelMarker(marker, label); // after construction: MapLibre sets its own generic label
        return marker;
      });
      note('');
    } catch (err) {
      $('#ov-weather').checked = false;
      note(`Weather layer unavailable: ${err.message}`);
      return;
    }
  }
  (weatherMarkers || []).forEach((m) => (m.getElement().style.display = on ? '' : 'none'));
}

async function setHotspots(on) {
  if (on && !hotspotMarkers) {
    try {
      const geojson = await getGeoJSON('/api/hotspots');
      hotspotMarkers = geojson.features.map((f) => {
        const p = f.properties;
        const popup = new maplibregl.Popup({ offset: 14, maxWidth: '280px' }).setDOMContent(
          h('div', {}, h('strong', {}, p.name), h('div', { class: 'muted small' }, p.province), h('p', {}, p.summary), h('p', { class: 'muted small' }, `Location: ${p.coordinate_basis}. Seed data: verify before relying on it.`)),
        );
        const el = h('div', { class: 'hotspot', title: p.name });
        const marker = new maplibregl.Marker({ element: el }).setLngLat(f.geometry.coordinates).setPopup(popup).addTo(map);
        labelMarker(marker, `Dioxin hotspot: ${p.name}`);
        return marker;
      });
    } catch (err) {
      $('#ov-hotspots').checked = false;
      note(`Hotspot layer unavailable: ${err.message}`);
      return;
    }
  }
  (hotspotMarkers || []).forEach((m) => (m.getElement().style.display = on ? '' : 'none'));
}

document.querySelectorAll('input[name=base]').forEach((r) => r.addEventListener('change', () => setBase(r.value)));
$('#ov-soil').addEventListener('change', (ev) => setSoil(ev.target.checked));
$('#ov-weather').addEventListener('change', (ev) => setWeather(ev.target.checked));
$('#ov-hotspots').addEventListener('change', (ev) => setHotspots(ev.target.checked));
setHotspots($('#ov-hotspots').checked);

map.on('click', (e) => {
  if (e.originalEvent.target.closest('.hotspot, .wx')) return; // markers have their own popups
  selectPoint(e.lngLat.lat, e.lngLat.lng);
});

// a shared link like #10.978,106.815 opens straight onto that point, on load and when the hash
// is edited or navigated to afterwards (selectPoint uses replaceState, which fires no event)
function openFromHash() {
  const linked = location.hash.match(/^#(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$/);
  if (linked) selectPoint(+linked[1], +linked[2], true);
}
window.addEventListener('hashchange', openFromHash);
openFromHash();
