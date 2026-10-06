/* Paired videos share physical cameras, frame count, frame rate and output size.
   Never substitute an existing Ours clip for missing baseline media. */
document.addEventListener('DOMContentLoaded', () => {
  const cards = [...document.querySelectorAll('[data-comparison-scene]')];
  const gallery = cards[0]?.parentElement;
  if (!gallery) return;
  const sceneOrder = ['skateboard', 'plant', 'bucket', 'flowerbed2'];
  cards.sort((a, b) => sceneOrder.indexOf(a.dataset.comparisonScene) - sceneOrder.indexOf(b.dataset.comparisonScene));
  cards.forEach(card => gallery.append(card));
  gallery.classList.add('removal-gallery');
  const tabs = document.createElement('div');
  tabs.className = 'removal-scenes';
  tabs.setAttribute('role', 'group');
  tabs.setAttribute('aria-label', 'Object removal scene');
  cards.forEach((card, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = card.querySelector('.scene-name-header').textContent;
    button.setAttribute('aria-pressed', String(index === 0));
    card.hidden = index !== 0;
    button.addEventListener('click', () => {
      cards.forEach(c => {c.hidden = c !== card; if (c.hidden) c.querySelectorAll('video').forEach(v => v.pause());});
      [...tabs.children].forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    });
    tabs.append(button);
  });
  gallery.before(tabs);
  document.querySelectorAll('[data-comparison-scene]').forEach(card => {
    const scene = card.dataset.comparisonScene;
    const root = `static/videos/comparison/${scene}/`;
    const title = card.querySelector('.scene-name-header').cloneNode(true);
    const original = document.createElement('div');
    original.className = 'removal-original';
    original.innerHTML = '<div class="wipe-labels"><span>Original scene</span></div>';
    const originals = [];
    for (const modality of ['rgb', 'depth']) {
      const pair = document.createElement('div');
      pair.className = 'wipe-pair';
      const video = document.createElement('video');
      video.src = `static/videos/${scene}_${modality}_ori.mp4`;
      video.muted = true; video.playsInline = true; video.loop = true; video.preload = 'metadata';
      video.setAttribute('aria-label', `${scene}, original, ${modality}`);
      const label = document.createElement('span');
      label.className = 'row-label'; label.textContent = modality === 'rgb' ? 'RGB' : 'Depth';
      pair.append(video, label); original.append(pair); originals.push(video);
    }
    const labels = document.createElement('div');
    labels.className = 'wipe-labels';
    labels.innerHTML = '<span>InstaInpaint</span><span>Ours (FreeInpaint)</span>';
    const stack = document.createElement('div');
    stack.className = 'wipe-stack';
    const videos = [];
    for (const modality of ['rgb', 'depth']) {
      const pair = document.createElement('div');
      pair.className = 'wipe-pair';
      for (const [method, side] of [['instainpaint', 'left'], ['ours', 'right']]) {
        const video = document.createElement('video');
        video.src = `${root}${method}_${modality}.mp4`;
        video.className = `wipe-${side}`;
        // Fixed display registration only; original media and camera poses are unchanged.
        if (scene === 'plant' && method === 'instainpaint') {
          video.style.transform = 'translateY(-3%)';
        }
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';
        video.setAttribute('aria-label', `${scene}, ${method}, ${modality}`);
        videos.push(video);
        pair.append(video);
      }
      const label = document.createElement('span');
      label.className = 'row-label';
      label.textContent = modality === 'rgb' ? 'RGB' : 'Depth';
      pair.append(label);
      stack.append(pair);
    }
    const line = document.createElement('div');
    line.className = 'wipe-line';
    line.setAttribute('aria-hidden', 'true');
    const range = document.createElement('input');
    Object.assign(range, {type: 'range', min: '0', max: '100', step: '.1', value: '50', className: 'wipe-range'});
    range.setAttribute('aria-label', `${scene}: adjust InstaInpaint / FreeInpaint split`);
    const update = () => {
      stack.style.setProperty('--split', `${range.value}%`);
      range.setAttribute('aria-valuetext', `${Math.round(range.value)}% InstaInpaint, ${Math.round(100 - range.value)}% FreeInpaint`);
    };
    range.addEventListener('input', update);
    // Native range preserves keyboard access; pointer mapping gives exact edge-to-edge positioning.
    const move = e => {
      const box = stack.getBoundingClientRect();
      range.value = Math.max(0, Math.min(100, (e.clientX-box.left)/box.width*100));
      update();
    };
    range.addEventListener('pointerdown', e => {range.setPointerCapture(e.pointerId); move(e);});
    range.addEventListener('pointermove', e => {if (range.hasPointerCapture(e.pointerId)) move(e);});
    range.addEventListener('pointerup', e => {if(range.hasPointerCapture(e.pointerId)) range.releasePointerCapture(e.pointerId);});
    stack.append(line, range);
    const controls = document.createElement('div');
    controls.className = 'wipe-controls';
    const play = document.createElement('button');
    Object.assign(play, {type: 'button', className: 'wipe-play', textContent: 'Play'});
    play.disabled = true;
    const hint = document.createElement('span');
    hint.className = 'wipe-hint'; hint.textContent = 'Drag to compare';
    controls.append(play, hint);
    const status = document.createElement('p');
    status.className = 'wipe-status'; status.textContent = 'Loading paired videos…';
    status.setAttribute('role', 'status');
    const comparison = document.createElement('div');
    comparison.className = 'removal-result';
    comparison.append(labels, stack);
    const columns = document.createElement('div');
    columns.className = 'removal-columns'; columns.append(original, comparison);
    const note = document.createElement('p');
    note.className = 'wipe-hint';
    note.textContent = scene === 'plant'
      ? 'Orbit: elevation −5°, radius 1.2. Approximate camera alignment; depth colors are normalized independently.'
      : 'Original: previous demo trajectory. Right: paired method comparison.';
    card.replaceChildren(title, columns, controls, status, note);
    update();
    let ready = false, playing = false, inView = false, userPaused = false, failed = false;
    const leader = videos[0];
    const pause = () => {playing = false; [...videos, ...originals].forEach(v => v.pause()); play.textContent = 'Play';};
    const start = async () => {
      if (!ready || failed || card.hidden) return;
      videos.slice(1).forEach(v => {v.currentTime = leader.currentTime;});
      try { await Promise.all(videos.map(v => v.play())); originals.forEach(v => v.play().catch(() => {})); playing = true; play.textContent = 'Pause'; }
      catch (_) { pause(); }
    };
    play.addEventListener('click', () => {userPaused = playing; if (playing) pause(); else start();});
    const error = message => {failed = true; pause(); play.disabled = true; status.textContent = message;};
    videos.forEach(v => {
      v.addEventListener('error', () => error('Paired video unavailable; please check the media files.'));
      v.addEventListener('loadedmetadata', () => {
        if (!videos.every(x => x.readyState >= 1) || ready || failed) return;
        const matched = videos.every(x => x.videoWidth === leader.videoWidth && x.videoHeight === leader.videoHeight && Math.abs(x.duration-leader.duration) < .02);
        if (!matched) return error('Video dimensions or durations differ; comparison paused.');
        ready = true; play.disabled = false; status.textContent = '';
        if (inView && !matchMedia('(prefers-reduced-motion: reduce)').matches) start();
      });
    });
    // One leader clock, restart all four together; no independently looping tracks.
    leader.addEventListener('ended', () => {videos.forEach(v => {v.currentTime = 0;}); if (playing) start();});
    const sync = () => {
      if (playing) videos.slice(1).forEach(v => {if (Math.abs(v.currentTime - leader.currentTime) > .025) v.currentTime = leader.currentTime;});
      if (leader.requestVideoFrameCallback) leader.requestVideoFrameCallback(sync);
    };
    if (leader.requestVideoFrameCallback) leader.requestVideoFrameCallback(sync);
    else leader.addEventListener('timeupdate', sync);
    originals[0].addEventListener('timeupdate', () => {
      if (originals[1].readyState >= 1 && Math.abs(originals[0].currentTime - originals[1].currentTime) > .06)
        originals[1].currentTime = originals[0].currentTime;
    });
    new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      if (!inView) pause(); else if (ready && !userPaused && !matchMedia('(prefers-reduced-motion: reduce)').matches) start();
    }, {threshold: .15}).observe(card);
  });
});
