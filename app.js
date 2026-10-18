(() => {
  'use strict';
  const CLASSES = ['bangtotnghiep','cccd','dtut','gxn','hocba10','hocba11','hocba12','hocbabia','vsat'];
  const LABELS = ['Bằng tốt nghiệp','CCCD','ĐTUT','GXN','Học bạ 10','Học bạ 11','Học bạ 12','Học bạ bìa','VSAT'];
  const MEAN = [0.485,0.456,0.406], STD = [0.229,0.224,0.225];
  const $ = (id) => document.getElementById(id);
  const state = { files: [], segmenter: null, classifier: null, rows: [], urls: [], busy: false };
  const status = (message) => { $('runStatus').textContent = message; };

  async function init() {
    try {
      if (!window.ort) throw new Error('Không tải được ONNX Runtime Web. Hãy mở trang khi có kết nối mạng hoặc kiểm tra thư mục vendor.');
      ort.env.wasm.wasmPaths = {
        mjs: new URL('vendor/ort-wasm-simd-threaded.mjs', document.baseURI).href,
        wasm: new URL('vendor/ort-wasm-simd-threaded.wasm', document.baseURI).href
      };
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      $('modelStatus').textContent = 'Đang tải ONNX…';
      const [segmenter, classifier] = await Promise.all([
        ort.InferenceSession.create('models/roi_segmenter.onnx', { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }),
        ort.InferenceSession.create('models/roi_classifier.onnx', { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
      ]);
      state.segmenter = segmenter;
      state.classifier = classifier;
      $('modelStatus').textContent = 'Mô hình đã sẵn sàng';
      $('modelStatus').classList.add('ready');
      $('runButton').disabled = state.files.length === 0;
      status('Đã tải hai nhánh ONNX. Lần chạy đầu có thể lâu hơn do trình duyệt khởi tạo WebAssembly.');
    } catch (error) {
      $('modelStatus').textContent = 'Không tải được model';
      status(error.message || String(error));
    }
  }

  function setFiles(files) {
    state.files = Array.from(files || []).filter(f => f.type.startsWith('image/'));
    $('runButton').disabled = !state.segmenter || state.files.length === 0 || state.busy;
    $('resultCount').textContent = state.files.length ? `${state.files.length} ảnh đã chọn` : 'Chưa có ảnh';
    status(state.files.length ? `Sẵn sàng nhận diện ${state.files.length} ảnh.` : 'Không tìm thấy ảnh hợp lệ.');
    $('progressBar').style.width = '0%';
  }

  function loadBitmap(file) {
    return createImageBitmap(file, { imageOrientation: 'from-image' });
  }

  function letterboxCanvas(bitmap, size, sourceBox = null, grayscale = false) {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff'; ctx.fillRect(0,0,size,size);
    let sx=0, sy=0, sw=bitmap.width, sh=bitmap.height;
    if (sourceBox) { [sx,sy,sw,sh] = sourceBox; }
    const scale = Math.min(size / sw, size / sh);
    const width = Math.max(1, Math.round(sw * scale)), height = Math.max(1, Math.round(sh * scale));
    const left = Math.floor((size-width)/2), top = Math.floor((size-height)/2);
    ctx.drawImage(bitmap, sx, sy, sw, sh, left, top, width, height);
    if (grayscale) {
      const pixels=ctx.getImageData(0,0,size,size);
      for(let p=0;p<pixels.data.length;p+=4){const l=0.299*pixels.data[p]+0.587*pixels.data[p+1]+0.114*pixels.data[p+2];pixels.data[p]=pixels.data[p+1]=pixels.data[p+2]=l;}
      ctx.putImageData(pixels,0,0);
    }
    return { canvas, scale, left, top };
  }

  function tensorFromCanvas(canvas, grayscale = false) {
    const { data } = canvas.getContext('2d', { willReadFrequently:true }).getImageData(0,0,canvas.width,canvas.height);
    const size = canvas.width * canvas.height, values = new Float32Array(3 * size);
    for (let p=0; p<size; p++) {
      const k=p*4; let r=data[k]/255, g=data[k+1]/255, b=data[k+2]/255;
      if (grayscale) { const l=0.299*r+0.587*g+0.114*b; r=g=b=l; }
      values[p]=(r-MEAN[0])/STD[0];
      values[size+p]=(g-MEAN[1])/STD[1];
      values[size*2+p]=(b-MEAN[2])/STD[2];
    }
    return new ort.Tensor('float32', values, [1,3,canvas.height,canvas.width]);
  }

  function findBox(maskData, bitmap, letterbox) {
    let minX=256,minY=256,maxX=-1,maxY=-1,count=0;
    for(let y=0;y<256;y++) for(let x=0;x<256;x++) {
      const v=maskData[y*256+x];
      if (1/(1+Math.exp(-v)) >= 0.45) { minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);count++; }
    }
    if(count<4) return { x:0,y:0,w:bitmap.width,h:bitmap.height, fallback:true };
    let x0=minX, y0=minY, x1=maxX+1, y1=maxY+1;
    const mx=Math.max(2,Math.round((x1-x0)*0.04)), my=Math.max(2,Math.round((y1-y0)*0.08));
    x0=Math.max(0,x0-mx); y0=Math.max(0,y0-my); x1=Math.min(256,x1+mx); y1=Math.min(256,y1+my);
    const {scale,left,top}=letterbox;
    const sx=Math.max(0,Math.min(bitmap.width,(x0-left)/scale));
    const sy=Math.max(0,Math.min(bitmap.height,(y0-top)/scale));
    const ex=Math.max(sx,Math.min(bitmap.width,(x1-left)/scale));
    const ey=Math.max(sy,Math.min(bitmap.height,(y1-top)/scale));
    return {x:sx,y:sy,w:Math.max(1,ex-sx),h:Math.max(1,ey-sy),fallback:false};
  }

  function softmax(logits) {
    let max=-Infinity; for(const v of logits) max=Math.max(max,v);
    const exps=Array.from(logits,v=>Math.exp(v-max)); const sum=exps.reduce((a,b)=>a+b,0);
    return exps.map(v=>v/sum);
  }

  async function predict(file) {
    const started=performance.now(), bitmap=await loadBitmap(file);
    try {
      const page=letterboxCanvas(bitmap,256);
      const maskOut=await state.segmenter.run({page:tensorFromCanvas(page.canvas)});
      const box=findBox(maskOut.mask_logits.data,bitmap,page);
      const crop=letterboxCanvas(bitmap,224,[box.x,box.y,box.w,box.h]);
      let out=await state.classifier.run({roi:tensorFromCanvas(crop.canvas)});
      let probabilities=softmax(out.class_logits.data);
      let predicted=probabilities.indexOf(Math.max(...probabilities));
      if (CLASSES[predicted] === 'cccd') {
        const grayPage=letterboxCanvas(bitmap,256,null,true);
        const grayMask=await state.segmenter.run({page:tensorFromCanvas(grayPage.canvas)});
        const grayBox=findBox(grayMask.mask_logits.data,bitmap,grayPage);
        const grayCrop=letterboxCanvas(bitmap,224,[grayBox.x,grayBox.y,grayBox.w,grayBox.h],true);
        out=await state.classifier.run({roi:tensorFromCanvas(grayCrop.canvas)});
        probabilities=softmax(out.class_logits.data);
        predicted=probabilities.indexOf(Math.max(...probabilities));
        box.grayRefined=true;
        Object.assign(box,grayBox);
      }
      const label=labelFromPath(file.webkitRelativePath || file.name);
      return { file, name:file.webkitRelativePath || file.name, trueLabel:label, predicted:CLASSES[predicted], confidence:probabilities[predicted], probabilities, box, ms:performance.now()-started, error:'' };
    } finally { bitmap.close(); }
  }

  function labelFromPath(path) {
    const parts=path.replaceAll('\\','/').split('/').map(x=>x.toLowerCase());
    for(let i=parts.length-1;i>=0;i--) { const found=CLASSES.indexOf(parts[i]); if(found>=0) return CLASSES[found]; }
    return '';
  }

  function drawResults() {
    $('resultCount').textContent=`${state.rows.length} ảnh đã chạy`;
    for(const url of state.urls) URL.revokeObjectURL(url);
    state.urls=[];
    const container=$('results'); container.replaceChildren();
    for(const row of state.rows) {
      const card=document.createElement('article'); card.className='item';
      if(row.trueLabel) card.classList.add(row.trueLabel===row.predicted?'correct':'wrong');
      const preview=document.createElement('div'); preview.className='preview';
      const img=document.createElement('img'); img.src=URL.createObjectURL(row.file); state.urls.push(img.src); img.alt=row.name; preview.append(img);
      const details=document.createElement('div'); details.className='details';
      const filename=document.createElement('div'); filename.className='filename'; filename.title=row.name; filename.textContent=row.name;
      const pred=document.createElement('div'); pred.className='pred'; pred.textContent=row.error?'Lỗi đọc ảnh':(LABELS[CLASSES.indexOf(row.predicted)]||row.predicted);
      const sub=document.createElement('div'); sub.className='sub';
      sub.textContent=row.error?row.error:`Độ tin cậy ${pct(row.confidence)} · ${row.ms.toFixed(1)} ms${row.box.grayRefined?' · đã xử lý xám CCCD':''}${row.box.fallback?' · dùng toàn ảnh':''}`;
      details.append(filename,pred,sub);
      if(row.trueLabel) { const badge=document.createElement('span'); badge.className='badge'; badge.textContent=`Nhãn thật: ${LABELS[CLASSES.indexOf(row.trueLabel)]}`; details.append(badge); }
      card.append(preview,details); container.append(card);
    }
    showSummary(); drawMatrix(); $('exportButton').disabled=state.rows.length===0;
  }

  function pct(value){return `${(value*100).toFixed(2)}%`;}
  function showSummary() {
    const valid=state.rows.filter(r=>!r.error), labelled=valid.filter(r=>r.trueLabel);
    const accuracy=labelled.length?labelled.filter(r=>r.trueLabel===r.predicted).length/labelled.length:null;
    const average=valid.length?valid.reduce((a,r)=>a+r.ms,0)/valid.length:null;
    const values=[['Ảnh xử lý',`${valid.length}/${state.rows.length}`],['Có nhãn thật',String(labelled.length)],['Accuracy',accuracy===null?'—':pct(accuracy)],['Thời gian TB',average===null?'—':`${average.toFixed(1)} ms`]];
    const box=$('summary'); box.replaceChildren(); box.hidden=valid.length===0;
    for(const [title,value] of values){const item=document.createElement('div');item.className='metric';const a=document.createElement('span');a.textContent=title;const b=document.createElement('strong');b.textContent=value;item.append(a,b);box.append(item);}
  }

  function drawMatrix() {
    const rows=state.rows.filter(r=>r.trueLabel&&!r.error);
    const card=$('matrixCard'); card.hidden=rows.length===0; if(!rows.length)return;
    const matrix=CLASSES.map(()=>CLASSES.map(()=>0));let correct=0;
    for(const row of rows){const a=CLASSES.indexOf(row.trueLabel),b=CLASSES.indexOf(row.predicted);if(a>=0&&b>=0){matrix[a][b]++;if(a===b)correct++;}}
    $('matrixAccuracy').textContent=`Accuracy ${pct(correct/rows.length)} · ${rows.length} ảnh có nhãn`;
    const table=$('matrixTable');table.replaceChildren();
    const head=document.createElement('tr');const corner=document.createElement('th');corner.textContent='Thật ↓ / Dự đoán →';head.append(corner);
    for(const label of LABELS){const th=document.createElement('th');th.textContent=label;head.append(th);}table.append(head);
    for(let i=0;i<CLASSES.length;i++){const tr=document.createElement('tr');const th=document.createElement('th');th.textContent=LABELS[i];tr.append(th);for(let j=0;j<CLASSES.length;j++){const td=document.createElement('td');td.textContent=matrix[i][j];if(i===j)td.className='diag';else if(matrix[i][j])td.className='off';tr.append(td);}table.append(tr);}
  }

  async function run() {
    if(state.busy||!state.files.length||!state.segmenter)return;
    state.busy=true; $('runButton').disabled=true;$('exportButton').disabled=true;state.rows=[];$('results').replaceChildren();$('summary').hidden=true;$('matrixCard').hidden=true;
    const files=state.files;let failed=0;
    try {
      for(let i=0;i<files.length;i++){
        status(`Đang chạy ONNX trong trình duyệt: ${i+1}/${files.length} · ${files[i].name}`);
        try{state.rows.push(await predict(files[i]));}catch(error){failed++;state.rows.push({file:files[i],name:files[i].webkitRelativePath||files[i].name,error:error.message||String(error),ms:0,box:{fallback:true}});}
        $('progressBar').style.width=`${((i+1)/files.length)*100}%`;
        if(i%3===0||i===files.length-1)drawResults();
        await new Promise(resolve=>setTimeout(resolve,0));
      }
      drawResults();status(`Hoàn tất ${files.length} ảnh${failed?` · lỗi ${failed}`:''}. Thời gian trên trình duyệt này được hiển thị riêng cho từng ảnh.`);
    } finally {state.busy=false;$('runButton').disabled=!state.segmenter||state.files.length===0;}
  }

  function exportCsv(){
    const cols=['file','true_label','predicted_label','confidence','inference_ms','roi_x','roi_y','roi_width','roi_height','error'];
    const quote=(x)=>`"${String(x??'').replaceAll('"','""')}"`;
    const lines=[cols.map(quote).join(',')];
    for(const r of state.rows)lines.push([r.name,r.trueLabel||'',r.predicted||'',r.confidence??'',r.ms?.toFixed(3)||'',r.box?.x??'',r.box?.y??'',r.box?.w??'',r.box?.h??'',r.error||''].map(quote).join(','));
    const blob=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='roi_predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  $('singleInput').addEventListener('change',e=>setFiles(e.target.files));
  $('folderInput').addEventListener('change',e=>setFiles(e.target.files));
  $('runButton').addEventListener('click',run);
  $('exportButton').addEventListener('click',exportCsv);
  init();
})();
