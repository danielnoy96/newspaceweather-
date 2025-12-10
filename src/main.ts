import { Pane } from 'tweakpane';

import { tcamera, mouse } from './input';
import renderShaders from './render.wgsl?raw';
import {
  hslToRgb,
  lerp5,
  linkRenderTimestamp,
  readTimestamp,
  requestTimestamps,
  resolveTimestamp,
  setupTimestamp,
} from './utils';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

const uniformsSize = 16;
const uniformData = new Float32Array(uniformsSize);

const simSize = 16;
const simData = new Float32Array(simSize);

const dt = 0.02;
const rMax = 15;
const forceFactor = 1;
const beta = 0.3;
const frictionHalfLife = 0.04;

function makeRandomMatrix() {
  const rows = [];
  for (let i = 0; i < colourAmt; i++) {
    const row = [];
    for (let j = 0; j < colourAmt; j++) {
      row.push(Math.random() * 2 - 1);
    }
    rows.push(row);
  }
  return rows;
}

let colourAmt = 200;
let colours: [number, number, number][] = [];
for (let i = 0; i < colourAmt; i++) {
  colours.push(hslToRgb((i / colourAmt) * 360, 1, 0.5));
}
let matrix = makeRandomMatrix();

const particleStride = 24;

const multistep = 1;

const cellAmt = 20000;

const camera = { ...tcamera };
const cameraData = new Float32Array([camera.x, camera.y, camera.zoom]);

let particleAmt = 50000;

let device: GPUDevice | undefined;
let context: GPUCanvasContext | undefined;
let uniformBuffer: GPUBuffer | undefined;
let simBuffer: GPUBuffer | undefined;
let renderPipeline: GPURenderPipeline | undefined;
let cameraBuffer: GPUBuffer | undefined;

let matrixBuffer: GPUBuffer | undefined;
let colourBuffer: GPUBuffer | undefined;

let particleBuffers: [GPUBuffer, GPUBuffer] | undefined;

let renderBindGroup: GPUBindGroup | undefined;

let alternate = 0;
let fpsc = 0;

const pane = new Pane({ title: 'GPU Life' });

const params = {
  fps: 0,
  engine: 'linkedList',
  particles: particleAmt,
  colours: colourAmt,
  cells: cellAmt,
};

pane.addBinding(params, 'fps', { readonly: true });

const engineSelect = pane.addBinding(params, 'engine', {
  options: {
    'Counting Sort': 'countingSort',
    'Atomic Linked Lists': 'linkedList',
    NSquared: 'nSquared',
  },
});

pane.addBinding(params, 'particles', { min: 1, step: 1 });

const newSimBtn = pane.addButton({ title: 'New Sim' });
const randomizeBtn = pane.addButton({ title: 'Randomize' });

const optionParams = {
  colours: colourAmt,
  r: rMax,
  force: forceFactor,
  beta: beta,
  delta: dt,
  friction: frictionHalfLife,
  cells: cellAmt,
  avoidance: 4,
  worldSize: 6,
  border: true,
  vortex: false,
};

const constantOptions = ['colours', 'cells'];

const constants: Record<string, BindingParams> = {
  colours: { min: 1, step: 1 },
  cells: { min: 1, step: 1 },
};

const constantsFolder = pane.addFolder({ title: 'Constants' });
for (const param in optionParams) {
  if (!constantOptions.includes(param)) continue;
  const binding = constantsFolder.addBinding(
    optionParams,
    param as keyof typeof optionParams,
    param in constants ? constants[param] : {},
  );
  binding.element.title = 'testing';
  binding.on('change', () => {
    setSim();
  });
}

const optionsFolder = pane.addFolder({ title: 'Options' });

function setSim() {
  if (!device || !simBuffer) return;
  simData[0] = colourAmt;
  simData[1] = optionParams.beta;
  simData[2] = 1 / optionParams.r;
  simData[3] = optionParams.force / (1 / optionParams.r);
  simData[4] = Math.pow(0.5, dt / optionParams.friction);
  simData[5] = optionParams.delta;
  simData[6] = (1 / optionParams.r) * 2;
  simData[7] = params.cells;
  simData[8] = optionParams.avoidance;
  simData[9] = optionParams.worldSize;
  simData[10] = optionParams.border ? 1 : 0;
  simData[11] = optionParams.vortex ? 1 : 0;
  device.queue.writeBuffer(simBuffer, 0, simData);
}

const options: Record<string, BindingParams> = {
  beta: { min: 0, max: 1 },
  r: { min: 0.01 },
  worldSize: { min: 0.01 },
  colours: { min: 1, step: 1 },
  cells: { min: 1, step: 1 },
};

for (const param in optionParams) {
  if (constantOptions.includes(param)) continue;
  const binding = optionsFolder.addBinding(
    optionParams,
    param as keyof typeof optionParams,
    param in options ? options[param] : {},
  );
  binding.element.title = 'testing';
  binding.on('change', () => {
    setSim();
  });
}

const performanceParams = {
  countingSort: {
    cell: 0,
    prefix: 0,
    sort: 0,
    sim: 0,
  },
  linkedList: {
    construct: 0,
    sim: 0,
  },
  nSquared: {
    sim: 0,
  },
};

const globalPerformanceParams = {
  cpu: 0,
  render: 0,
  total: 0,
  graph: 0,
};

const performanceTimes = pane.addFolder({ title: 'Performance Times' });
const engineFolders: Record<string, FolderApi> = {};

// folder.addBinding(performanceParams, 'countingSort');

for (const engine in performanceParams) {
  engineFolders[engine] = performanceTimes.addFolder({
    title: engine,
    hidden: true,
  });

  const params = performanceParams[engine as keyof typeof performanceParams];

  for (const param in params) {
    engineFolders[engine].addBinding(params, param as keyof typeof params, {
      readonly: true,
      interval: 50,
      format: (v: number) => `${v.toFixed(2)}ms`,
    });
  }
}

// const bindings: Record<string, BindingApi> = {};

for (const param in globalPerformanceParams) {
  if (param == 'graph') continue;
  performanceTimes.addBinding(
    globalPerformanceParams,
    param as keyof typeof globalPerformanceParams,
    {
      readonly: true,
      interval: 50,
      format: (v: number) => `${v.toFixed(2)}ms`,
    },
  );
}

performanceTimes.addBinding(globalPerformanceParams, 'graph', {
  label: '',
  readonly: true,
  view: 'graph',
  min: 0,
  max: 1,
  interval: 50,
});

function setEngineDisplay(engine: string) {
  for (const engine2 in performanceParams) {
    engineFolders[engine2].hidden = engine2 != engine;
  }
}

import * as nSquared from './nSquared/main';
import * as linkedList from './linkedList/main';
import * as countingSort from './countingSort/main';
import { type BindingParams, type FolderApi } from '@tweakpane/core';

const engines: Record<
  string,
  typeof nSquared | typeof linkedList | typeof countingSort
> = {
  nSquared,
  linkedList,
  countingSort,
};

let engine: string = 'linkedList';
setEngineDisplay(engine);

// const test = {
//   one: 123,
//   two: 345,
//   three: 'testing123',
// };

// const gui = new GUI();
// const cubeFolder = gui.addFolder('Cube');
// cubeFolder.add(test, 'one');
// cubeFolder.add(test, 'two');
// cubeFolder.add(test, 'three');
// cubeFolder.open();

(async () => {
  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'compatibility',
  });

  if (!adapter) return;

  device = await requestTimestamps(adapter);
  context = canvas.getContext('webgpu') ?? undefined;

  if (!context) return;

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({ device, format: presentationFormat });

  //

  uniformBuffer = device.createBuffer({
    size: uniformsSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'uniformBuffer',
  });

  simBuffer = device.createBuffer({
    size: simSize * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'simBuffer',
  });

  cameraBuffer = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'cameraBuffer',
  });

  //

  for (const engine in engines) {
    engines[engine].setup(device);
  }

  const renderModule = device.createShaderModule({ code: renderShaders });

  renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vertex',
      buffers: [
        {
          arrayStride: particleStride,
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 8, format: 'float32x2' },
            { shaderLocation: 2, offset: 16, format: 'float32' },
          ],
        },
      ],
    },
    fragment: {
      module: renderModule,
      entryPoint: 'fragment',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-strip',
    },
  });

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  setupTimestamp(device, 'render');

  startParticles();
})();

function tick(commandEncoder: GPUCommandEncoder) {
  if (!device) return;
  engines[engine].tick(device, commandEncoder, alternate, particleAmt);
  alternate = (alternate + 1) % 2;
}

function render(context: GPUCanvasContext, commandEncoder: GPUCommandEncoder) {
  if (!renderPipeline || !particleBuffers || !device) return;

  const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

  const passEncoder = commandEncoder.beginRenderPass(
    linkRenderTimestamp(device, renderPassDescriptor, 'render'),
  );
  passEncoder.setPipeline(renderPipeline);
  passEncoder.setVertexBuffer(0, particleBuffers[(alternate + 1) % 2]);
  passEncoder.setBindGroup(0, renderBindGroup);
  passEncoder.draw(6, particleAmt, 0, 0);
  passEncoder.end();

  resolveTimestamp(device, commandEncoder, 'render');
}

function startParticles() {
  if (
    !device ||
    !uniformBuffer ||
    !renderPipeline ||
    !simBuffer ||
    !cameraBuffer
  )
    return;

  const bufferSize = particleAmt * particleStride;
  particleBuffers = [
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
      size: bufferSize,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST,
    }),
  ];

  alternate = 0;
  const data = new Float32Array(bufferSize / 4);
  let pi = 0;
  while (pi < particleAmt) {
    const spawnAmt = ((Math.random() * (particleAmt - pi)) / colourAmt) * 5;
    const c = Math.floor(Math.random() * colourAmt);

    const a = Math.random() * Math.PI * 2;
    const d = Math.random() * optionParams.worldSize * 0.9;

    const x = Math.cos(a) * d;
    const y = Math.sin(a) * d;
    for (let i = 0; i < spawnAmt; i++) {
      const a = Math.random() * Math.PI * 2;
      const d = (Math.random() ** 3 / 10) * optionParams.worldSize;
      data[pi * 6] = x + Math.cos(a) * d;
      data[pi * 6 + 1] = y + Math.sin(a) * d;
      data[pi * 6 + 2] = 0;
      data[pi * 6 + 3] = 0;
      data[pi * 6 + 4] = c;

      data[pi * 6 + 5] = 0;

      pi++;
    }
  }
  // for (let i = 0; i < particleAmt; i++) {
  //   data[i * 6] = Math.random() * 2 - 1;
  //   data[i * 6 + 1] = Math.random() * 2 - 1;
  //   data[i * 6 + 2] = 0;
  //   data[i * 6 + 3] = 0;
  //   data[i * 6 + 4] = Math.floor(Math.random() * colourAmt);

  //   data[i * 6 + 5] = 0;
  //   // data[i * 8 + 6] = 0;
  //   // data[i * 8 + 7] = 0;
  // }

  device.queue.writeBuffer(particleBuffers[0], 0, data.buffer);

  setSim();

  matrixBuffer = device.createBuffer({
    size: colourAmt * colourAmt * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const matrixData = new Float32Array(colourAmt * colourAmt);
  for (let c1 = 0; c1 < colourAmt; c1++) {
    for (let c2 = 0; c2 < colourAmt; c2++) {
      matrixData[c1 * colourAmt + c2] = matrix[c1][c2];
    }
  }
  device.queue.writeBuffer(matrixBuffer, 0, matrixData.buffer);

  colourBuffer = device.createBuffer({
    size: colourAmt * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const colourData = new Float32Array(colourAmt * 3);
  for (let c = 0; c < colourAmt; c++) {
    colourData[c * 3] = colours[c][0];
    colourData[c * 3 + 1] = colours[c][1];
    colourData[c * 3 + 2] = colours[c][2];
  }
  device.queue.writeBuffer(colourBuffer, 0, colourData.buffer);

  for (const engine in engines) {
    engines[engine].start(
      device,
      uniformBuffer,
      simBuffer,
      matrixBuffer,
      particleBuffers,
      particleAmt,
      cellAmt,
    );
  }

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: uniformBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: cameraBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: colourBuffer,
        },
      },
    ],
  });
}

let lastTime = 0;
const deltaVs: number[] = [];

function update() {
  requestAnimationFrame(update);
  if (!device || !context) return;

  const start = performance.now();
  const delta = (start - lastTime) / 1000;
  deltaVs.push(start - lastTime);
  if (deltaVs.length > 1000) {
    deltaVs.splice(0, 1);
  }
  lastTime = start;

  camera.x = lerp5(camera.x, tcamera.x, delta * 50);
  camera.y = lerp5(camera.y, tcamera.y, delta * 50);
  camera.zoom = lerp5(camera.zoom, tcamera.zoom, delta * 50);

  const commandEncoder = device.createCommandEncoder();

  if (uniformBuffer) {
    // uniformData[0] = canvas.width;
    // uniformData[1] = canvas.height;
    // uniformData[2] = Math.floor(Math.random() * 10000);

    uniformData[0] = canvas.width / canvas.height;

    uniformData[4] = mouse.x;
    uniformData[5] = mouse.y;
    uniformData[6] = mouse.down;
    uniformData[7] = mouse.type;

    uniformData[8] = (1 / optionParams.r) * 0.015;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  }

  if (cameraBuffer) {
    cameraData[0] = camera.x;
    cameraData[1] = camera.y;
    cameraData[2] = camera.zoom;

    device.queue.writeBuffer(cameraBuffer, 0, cameraData);
  }

  for (let i = 0; i < multistep; i++) {
    tick(commandEncoder);
  }

  render(context, commandEncoder);

  const commands = commandEncoder.finish();
  device.queue.submit([commands]);

  device.popErrorScope().then((error) => {
    if (error) {
      // some weird bug happened with timestamps, just disable it and restart the simulation
      const url = new URL(window.location.href);
      url.searchParams.set('noTimestamp', 'true');
      window.location.href = url.toString();
    }
  });

  const cpuTime = performance.now() - start;
  globalPerformanceParams.cpu = cpuTime;
  updateTotal();

  for (const engine2 in engines) {
    if (engine == engine2) {
      engines[engine2].updateDisplays(
        performanceParams[engine2 as keyof typeof performanceParams],
      );
    }
  }

  readTimestamp('render').then((time) => {
    globalPerformanceParams.render = time;
    updateTotal();
  });

  // let timeContent = '';
  // for (const sim in times) {
  //   for (const time in times[sim]) {
  //     if (times[sim][time]) {
  //       timeContent += `${time}: ${times[sim][time]} <br>`;
  //     }
  //   }
  // }

  // timesDisplay.innerHTML = timeContent;

  fpsc++;
}

requestAnimationFrame(update);
// setInterval(update, 1000 / 60);

// const particleAmtI = document.getElementById(
//   'particle-amount',
// ) as HTMLInputElement;

// particleAmtI.value = particleAmt + '';

newSimBtn.on('click', () => {
  colourAmt = optionParams.colours;
  colours = [];
  for (let i = 0; i < colourAmt; i++) {
    colours.push(hslToRgb((i / colourAmt) * 360, 1, 0.5));
  }

  params.cells = optionParams.cells;
  particleAmt = params.particles;

  matrix = makeRandomMatrix();
  startParticles();
});

randomizeBtn.on('click', () => {
  if (!device || !matrixBuffer) return;
  matrix = makeRandomMatrix();
  const matrixData = new Float32Array(colourAmt * colourAmt);
  for (let c1 = 0; c1 < colourAmt; c1++) {
    for (let c2 = 0; c2 < colourAmt; c2++) {
      matrixData[c1 * colourAmt + c2] = matrix[c1][c2];
    }
  }
  device.queue.writeBuffer(matrixBuffer, 0, matrixData.buffer);
});

// const fpsDisplay = document.getElementById('fps') as HTMLHeadingElement;

setInterval(() => {
  params.fps = fpsc;
  // fpsDisplay.textContent = `FPS: ${fpsc}`;
  fpsc = 0;
}, 1000);

engineSelect.on('change', (event) => {
  engine = event.value;
  setEngineDisplay(engine);
  // for (const sim in times) {
  //   if (sim == 'general') continue;
  //   for (const time in times[sim]) {
  //     times[sim][time] = engine == sim ? 'a' : '';
  //   }
  //   if (engine != sim) {
  //     engines[sim].cancelDisplays();
  //   }
  // }
});

function updateTotal() {
  let deltaTotal = 0;
  for (const time of deltaVs) {
    deltaTotal += time;
  }
  deltaTotal /= deltaVs.length;
  if (isNaN(deltaTotal)) deltaTotal = 0;

  let total = 0;
  const params = performanceParams[engine as keyof typeof performanceParams];
  for (const pass in params) {
    total += params[pass as keyof typeof params];
  }
  total += globalPerformanceParams.cpu;
  total += globalPerformanceParams.render;
  globalPerformanceParams.total = total;
  globalPerformanceParams.graph = total / deltaTotal;
}

window.onresize = () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
};
