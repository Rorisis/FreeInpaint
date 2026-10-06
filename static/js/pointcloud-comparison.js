/* Local WebGL point viewer. No image-space warping or invented demo geometry. */
document.addEventListener('DOMContentLoaded', () => {
  const host = document.getElementById('wild-clouds');
  if (!host) return;
  const scenes = [
    {id: 'co3d_bottle_575_84616_167245', title: 'Bottle', dataset: 'CO3D'},
    {id: 'co3d_cup_14_202_1205', title: 'Cup', dataset: 'CO3D'},
    {id:'co3d_bench_106_12654_23221',title:'Bench',dataset:'CO3D',defaultDistance:2.2},
    {id: 'colema_statue', title: 'Colema statue'},
    {id: 'bagpack_laptop_cup', title: 'Backpack, laptop & cup'},
    {id: 'bear_and_girl_statue', title: 'Bear & girl statue'},
  ];
  if(host.dataset.variant==='donut-v2') scenes.push(
    {id:'co3d_donut_426_59801_116462',title:'Donut',dataset:'CO3D'}
  );
  host.innerHTML = `<h3 class="subsection-title">Additional In-the-Wild Scenes</h3>
    <div class="cloud-example-badge">Interactive Examples</div>
    <div class="cloud-scenes" role="group" aria-label="Interactive 3D scene"></div>
    <div class="cloud-row">
      <div class="cloud-panel cloud-inputs"><h4>Input images<small>8 shared views</small></h4>
        <img class="cloud-cover" alt="Inpainted reference image">
        <details><summary>Show all 8 inputs</summary><div class="cloud-input-grid"></div></details>
      </div>
      ${[['instainpaint','InstaInpaint','with COLMAP'],['ours','Ours (FreeInpaint)','without COLMAP']].map(([id,title,subtitle]) => `<div class="cloud-panel" data-cloud-method="${id}"><h4>${title}<small>${subtitle}</small></h4><canvas class="cloud-canvas" tabindex="0" aria-label="${title}: drag to rotate, scroll to zoom, arrow keys to rotate"></canvas><div class="cloud-meta"><span>Loading…</span><a download>Download PLY</a></div></div>`).join('')}
    </div>
    <div class="cloud-controls"><button type="button">Reset view</button><label>Point size <input type="range" min="1" max="5" step=".25" value="2" aria-label="Point size"></label><span>Drag to rotate · scroll / pinch to zoom · Shift-drag to pan</span></div>
    <p class="cloud-status" role="status"></p>
    <p class="cloud-note">Colored Gaussian centers, not splat rendering. Both viewers use the same opacity threshold and point budget; each cloud is centered and scaled for display. Inputs include one inpainted reference and seven masked views. InstaInpaint uses dataset cameras (COLMAP for GS25; official annotations for CO3D). FreeInpaint predicts its cameras from images.</p>`;
  const status = host.querySelector('.cloud-status');
  const assetRoot = scene => {
    let folder='pointclouds';
    if(scene==='bagpack_laptop_cup') folder='pointclouds-cup';
    if(scene==='co3d_bench_106_12654_23221') folder='pointclouds-bench-refined';
    if(host.dataset.variant==='donut-v2' && scene==='co3d_donut_426_59801_116462') folder='pointclouds-donut-v2';
    if(host.dataset.variant==='car-bench' && ['co3d_car_106_12650_23736','co3d_bench_106_12654_23221'].includes(scene)) folder='pointclouds-car-bench';
    if(host.dataset.variant==='bench-refined' && scene==='co3d_bench_106_12654_23221') folder='pointclouds-bench-refined';
    return `static/${folder}/${scene}/`;
  };
  let state = {yaw: 0, pitch: 0, distance: 1, pan: [0, 0], size: 2};
  const viewers = [];
  const shader = (gl, kind, source) => {
    const s = gl.createShader(kind); gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s));
    return s;
  };
  function createViewer(canvas) {
    const gl = canvas.getContext('webgl', {antialias:true, alpha:false});
    if (!gl) throw Error('WebGL is unavailable. Please enable hardware acceleration or download the PLY.');
    const program = gl.createProgram();
    gl.attachShader(program, shader(gl, gl.VERTEX_SHADER, `attribute vec3 position; attribute vec3 color; varying vec3 rgb;
      uniform vec2 angles; uniform vec2 pan; uniform vec3 camera; uniform float distance; uniform float aspect; uniform float size;
      void main(){ float cy=cos(angles.x),sy=sin(angles.x),cx=cos(angles.y),sx=sin(angles.y);
        vec3 p=vec3(cy*position.x+sy*position.z,position.y,-sy*position.x+cy*position.z);
        p=vec3(p.x,cx*p.y-sx*p.z,sx*p.y+cx*p.z); p+=camera*distance; p.xy+=pan; float z=p.z;
        gl_Position=vec4(1.7*p.x/aspect,-1.7*p.y,1.000002*z-.0002,z);
        gl_PointSize=size; rgb=color; }`));
    gl.attachShader(program, shader(gl, gl.FRAGMENT_SHADER, `precision mediump float; varying vec3 rgb;
      void main(){if(length(gl_PointCoord-vec2(.5))>.5) discard; gl_FragColor=vec4(rgb,1.);}`));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
    const buffers = [gl.createBuffer(),gl.createBuffer()];
    let count = 0, camera = [0,0,2.8];
    function draw() {
      const ratio = Math.min(devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(canvas.clientWidth*ratio)), h = Math.max(1, Math.round(canvas.clientHeight*ratio));
      if (canvas.width !== w || canvas.height !== h) {canvas.width=w; canvas.height=h;}
      gl.viewport(0,0,w,h); gl.clearColor(.96,.97,.985,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      if (!count) return;
      gl.enable(gl.DEPTH_TEST); gl.useProgram(program);
      ['position','color'].forEach((name,i) => {
        const loc = gl.getAttribLocation(program,name); gl.bindBuffer(gl.ARRAY_BUFFER,buffers[i]); gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc,3,i ? gl.UNSIGNED_BYTE : gl.FLOAT,Boolean(i),0,0);
      });
      gl.uniform2f(gl.getUniformLocation(program,'angles'),state.yaw,state.pitch);
      gl.uniform2f(gl.getUniformLocation(program,'pan'),...state.pan);
      gl.uniform3f(gl.getUniformLocation(program,'camera'),...camera);
      gl.uniform1f(gl.getUniformLocation(program,'distance'),state.distance);
      gl.uniform1f(gl.getUniformLocation(program,'aspect'),w/h);
      gl.uniform1f(gl.getUniformLocation(program,'size'),state.size*ratio);
      gl.drawArrays(gl.POINTS,0,count);
    }
    canvas.addEventListener('webglcontextlost', e => {e.preventDefault(); status.textContent='Graphics context lost. Reload this page to restore the viewers.';});
    return {canvas, draw, load(data) {
      [data.xyz,data.rgb].forEach((encoded,i) => {
        const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
        gl.bindBuffer(gl.ARRAY_BUFFER,buffers[i]); gl.bufferData(gl.ARRAY_BUFFER,bytes,gl.STATIC_DRAW);
      }); count=data.count; camera=data.camera || [0,0,2.8]; draw(); canvas.dataset.points=String(count);
    }, clear() { count=0; draw(); }};
  }
  try {host.querySelectorAll('canvas').forEach(c => viewers.push(createViewer(c)));}
  catch(e) {status.textContent=e.message; return;}
  const draw = () => viewers.forEach(v=>v.draw());
  const reset = () => {const scene=scenes.find(s=>s.id===host.dataset.scene);state={...state,yaw:0,pitch:0,distance:scene?.defaultDistance || 1,pan:[0,0]};draw();};
  viewers.forEach(({canvas})=>{
    const pointers=new Map();
    canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);});
    canvas.addEventListener('pointermove',e=>{
      if(!pointers.has(e.pointerId))return;
      const previous=pointers.get(e.pointerId), next=[e.clientX,e.clientY];
      if(pointers.size===2){ const other=[...pointers.entries()].find(([id])=>id!==e.pointerId)[1];
        const old=Math.hypot(previous[0]-other[0],previous[1]-other[1]), now=Math.hypot(next[0]-other[0],next[1]-other[1]);
        if(now>1) state.distance=Math.max(.3,Math.min(15,state.distance*old/now));
      }else if(e.shiftKey){state.pan[0]+=(next[0]-previous[0])*.004;state.pan[1]+=(next[1]-previous[1])*.004;}
      else {state.yaw+=(next[0]-previous[0])*.008;state.pitch+=(next[1]-previous[1])*.008;}
      pointers.set(e.pointerId,next);draw();
    });
    ['pointerup','pointercancel','lostpointercapture'].forEach(type=>canvas.addEventListener(type,e=>pointers.delete(e.pointerId)));
    canvas.addEventListener('wheel',e=>{e.preventDefault();state.distance=Math.max(.3,Math.min(15,state.distance*Math.exp(e.deltaY*.001)));draw();},{passive:false});
    canvas.addEventListener('keydown',e=>{
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','0'].includes(e.key))return;e.preventDefault();
      if(e.key==='0')return reset();
      if(e.key==='ArrowLeft')state.yaw-=.1;if(e.key==='ArrowRight')state.yaw+=.1;
      if(e.key==='ArrowUp')state.pitch-=.1;if(e.key==='ArrowDown')state.pitch+=.1;
      if(e.key==='+'||e.key==='=')state.distance=Math.max(.3,state.distance*.9);if(e.key==='-')state.distance=Math.min(15,state.distance*1.1);draw();
    });
  });
  new ResizeObserver(draw).observe(host);
  host.querySelector('.cloud-controls button').addEventListener('click',reset);
  host.querySelector('input[type=range]').addEventListener('input',e=>{state.size=Number(e.target.value);draw();});
  const requests=new Map();
  function loadData(scene,method) {
    const key=scene+'/'+method;
    if(!requests.has(key)) requests.set(key,new Promise((resolve,reject)=>{
      const script=document.createElement('script'); script.src=assetRoot(scene)+method+'.js';
      script.onload=()=>{const data=window.FreeInpaintClouds?.[key];data?resolve(data):reject(Error('Invalid point-cloud data'));};
      script.onerror=()=>{requests.delete(key);reject(Error('Point-cloud file unavailable.'));};document.head.append(script);
    }));
    return requests.get(key);
  }
  let generation=0;
  async function selectScene(scene) {
    const current=++generation;host.dataset.scene=scene.id;status.textContent='Loading point clouds…';viewers.forEach(v=>v.clear());reset();
    host.querySelectorAll('.cloud-scenes button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scene===scene.id)));
    host.querySelector('details').open=false;
    host.querySelector('[data-cloud-method="instainpaint"] h4 small').textContent=scene.dataset==='CO3D' ? 'with official CO3D cameras' : 'with COLMAP';
    host.querySelector('[data-cloud-method="ours"] h4 small').textContent=scene.dataset==='CO3D' ? 'without camera poses' : 'without COLMAP';
    const root=assetRoot(scene.id);
    host.querySelector('.cloud-cover').src=root+'input_00.jpg';
    host.querySelector('.cloud-input-grid').innerHTML=Array.from({length:8},(_,i)=>`<figure><img loading="lazy" src="${root}input_${String(i).padStart(2,'0')}.jpg" alt="${scene.title}, ${i===0?'inpainted reference':'masked source '+i}"><figcaption>${i===0?'Reference':'View '+(i+1)}</figcaption></figure>`).join('');
    host.querySelectorAll('[data-cloud-method]').forEach(panel=>{panel.querySelector('.cloud-meta span').textContent='Loading…';panel.querySelector('a').href=root+panel.dataset.cloudMethod+'.ply';});
    try {
      const data=await Promise.all(['instainpaint','ours'].map(m=>loadData(scene.id,m)));
      if(current!==generation)return;
      data.forEach((d,i)=>{viewers[i].load(d);viewers[i].canvas.parentElement.querySelector('.cloud-meta span').textContent=d.count.toLocaleString()+' points';});
      status.textContent=(scene.dataset || 'GS25')+' · linked controls';
    }catch(e){if(current===generation)status.textContent=e.message;}
  }
  scenes.forEach(scene=>{const b=document.createElement('button');b.type='button';b.textContent=scene.title;b.dataset.scene=scene.id;b.addEventListener('click',()=>selectScene(scene));host.querySelector('.cloud-scenes').append(b);});
  // Load only when approaching the interactive section.
  const initialScene=['car-bench','bench-refined'].includes(host.dataset.variant) ? scenes.find(s=>s.id==='co3d_bench_106_12654_23221') : host.dataset.variant==='donut-v2' ? scenes.find(s=>s.id==='co3d_donut_426_59801_116462') : scenes[0];
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){selectScene(initialScene);observer.disconnect();}},{rootMargin:'300px'});
  observer.observe(host);
});
