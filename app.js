(() => {
  'use strict';
  const CLASSES = ['bangtotnghiep','cccd','dtut','gxn','hocba10','hocba11','hocba12','hocbabia','vsat'];
  const LABELS = ['Bằng tốt nghiệp','CCCD','ĐTUT','GXN','Học bạ 10','Học bạ 11','Học bạ 12','Học bạ bìa','VSAT'];
  const MEAN = [0.485,0.456,0.406], STD = [0.229,0.224,0.225];
  const $ = id => document.getElementById(id);
  const state = {files:[], segmenter:null, classifier:null, rows:[], urls:[], busy:false, modelLoadMs:0};
  const status = message => { $('runStatus').textContent=message; };

  async function init(){
    const started=performance.now();
    try{
      if(!window.ort) throw new Error('Không tải được ONNX Runtime.');
      ort.env.wasm.wasmPaths={mjs:new URL('vendor/ort-wasm-simd-threaded.mjs',document.baseURI).href,wasm:new URL('vendor/ort-wasm-simd-threaded.wasm',document.baseURI).href};
      ort.env.wasm.numThreads=1; ort.env.wasm.proxy=false;
      const [segmenter,classifier]=await Promise.all([
        ort.InferenceSession.create('models/roi_segmenter.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'all'}),
        ort.InferenceSession.create('models/roi_classifier.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'all'})
      ]);
      state.segmenter=segmenter; state.classifier=classifier; state.modelLoadMs=performance.now()-started;
      $('modelStatus').textContent=`Sẵn sàng · tải ${state.modelLoadMs.toFixed(0)} ms`;
      $('runButton').disabled=state.files.length===0;
      status('Chọn ảnh hoặc thư mục.');
    }catch(error){$('modelStatus').textContent='Lỗi tải model';status(error.message||String(error));}
  }

  function setFiles(files){
    state.files=Array.from(files||[]).filter(file=>file.type.startsWith('image/'));
    $('runButton').disabled=!state.segmenter||!state.files.length||state.busy;
    $('resultCount').textContent=state.files.length?`${state.files.length} ảnh đã chọn`:'Chưa có ảnh';
    status(state.files.length?`Đã chọn ${state.files.length} ảnh.`:'Không có ảnh hợp lệ.');
    $('progressBar').style.width='0%';
  }

  const loadBitmap=file=>createImageBitmap(file,{imageOrientation:'from-image'});
  function letterboxCanvas(bitmap,size,sourceBox=null){
    const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.fillStyle='#fff';ctx.fillRect(0,0,size,size);
    let sx=0,sy=0,sw=bitmap.width,sh=bitmap.height;if(sourceBox)[sx,sy,sw,sh]=sourceBox;
    const scale=Math.min(size/sw,size/sh),width=Math.max(1,Math.round(sw*scale)),height=Math.max(1,Math.round(sh*scale));
    const left=Math.floor((size-width)/2),top=Math.floor((size-height)/2);
    ctx.drawImage(bitmap,sx,sy,sw,sh,left,top,width,height);
    return {canvas,scale,left,top};
  }
  function tensorFromCanvas(canvas){
    const {data}=canvas.getContext('2d',{willReadFrequently:true}).getImageData(0,0,canvas.width,canvas.height);
    const size=canvas.width*canvas.height,values=new Float32Array(3*size);
    for(let p=0;p<size;p++){
      const k=p*4,r=data[k]/255,g=data[k+1]/255,b=data[k+2]/255;
      values[p]=(r-MEAN[0])/STD[0];values[size+p]=(g-MEAN[1])/STD[1];values[size*2+p]=(b-MEAN[2])/STD[2];
    }
    return new ort.Tensor('float32',values,[1,3,canvas.height,canvas.width]);
  }
  function findBox(maskData,bitmap,lb){
    let minX=256,minY=256,maxX=-1,maxY=-1,count=0;
    for(let y=0;y<256;y++)for(let x=0;x<256;x++)if(1/(1+Math.exp(-maskData[y*256+x]))>=0.45){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);count++;}
    if(count<4)return{x:0,y:0,w:bitmap.width,h:bitmap.height,fallback:true,maskBox:[0,0,256,256],activePixels:count};
    let x0=minX,y0=minY,x1=maxX+1,y1=maxY+1;
    const mx=Math.max(2,Math.round((x1-x0)*0.04)),my=Math.max(2,Math.round((y1-y0)*0.08));
    x0=Math.max(0,x0-mx);y0=Math.max(0,y0-my);x1=Math.min(256,x1+mx);y1=Math.min(256,y1+my);
    const {scale,left,top}=lb,sx=Math.max(0,Math.min(bitmap.width,(x0-left)/scale)),sy=Math.max(0,Math.min(bitmap.height,(y0-top)/scale));
    const ex=Math.max(sx,Math.min(bitmap.width,(x1-left)/scale)),ey=Math.max(sy,Math.min(bitmap.height,(y1-top)/scale));
    return{x:sx,y:sy,w:Math.max(1,ex-sx),h:Math.max(1,ey-sy),fallback:false,maskBox:[x0,y0,x1,y1],activePixels:count};
  }
  function softmax(logits){let max=-Infinity;for(const x of logits)max=Math.max(max,x);const e=Array.from(logits,x=>Math.exp(x-max)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);}
  function tensorStats(values){let min=Infinity,max=-Infinity,sum=0;for(const value of values){min=Math.min(min,value);max=Math.max(max,value);sum+=value;}return{count:values.length,min,max,mean:sum/values.length};}
  function overlayUrl(pageCanvas,maskData,box){
    const canvas=document.createElement('canvas');canvas.width=canvas.height=256;const ctx=canvas.getContext('2d');ctx.drawImage(pageCanvas,0,0);
    const image=ctx.getImageData(0,0,256,256);
    for(let i=0;i<maskData.length;i++){const p=1/(1+Math.exp(-maskData[i]));if(p>=0.45){const k=i*4;image.data[k]=Math.round(image.data[k]*0.38+255*0.62);image.data[k+1]=Math.round(image.data[k+1]*0.38+42*0.62);image.data[k+2]=Math.round(image.data[k+2]*0.38+42*0.62);}}
    ctx.putImageData(image,0,0);
    const [x0,y0,x1,y1]=box.maskBox;ctx.strokeStyle='#00e676';ctx.lineWidth=2;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    return canvas.toDataURL('image/png');
  }
  async function predict(file){
    const started=performance.now(),bitmap=await loadBitmap(file);
    try{
      const page=letterboxCanvas(bitmap,256);
      const maskOut=await state.segmenter.run({page:tensorFromCanvas(page.canvas)});
      const mask=maskOut.mask_logits.data,box=findBox(mask,bitmap,page),maskStats=tensorStats(mask);
      const crop=letterboxCanvas(bitmap,224,[box.x,box.y,box.w,box.h]);
      const classifierOut=await state.classifier.run({roi:tensorFromCanvas(crop.canvas)});
      const logits=Array.from(classifierOut.class_logits.data),probabilities=softmax(logits);
      const predicted=probabilities.indexOf(Math.max(...probabilities));
      return{file,name:file.webkitRelativePath||file.name,trueLabel:labelFromPath(file.webkitRelativePath||file.name),predicted:CLASSES[predicted],confidence:probabilities[predicted],probabilities,logits,maskLogits:Array.from(mask),maskStats,box,ms:performance.now()-started,overlay:overlayUrl(page.canvas,mask,box),roiImage:crop.canvas.toDataURL('image/jpeg',0.92),error:''};
    }finally{bitmap.close();}
  }
  function labelFromPath(path){const parts=path.replaceAll('\\','/').split('/').map(x=>x.toLowerCase());for(let i=parts.length-1;i>=0;i--){const found=CLASSES.indexOf(parts[i]);if(found>=0)return CLASSES[found];}return'';}
  const pct=x=>`${(x*100).toFixed(2)}%`,fmt=x=>Number(x).toFixed(5);

  function renderScores(row){
    const table=document.createElement('table');table.className='score-table';
    const head=document.createElement('tr');for(const name of ['ID','Lớp','Logit','Softmax']){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
    CLASSES.forEach((name,i)=>{const tr=document.createElement('tr');if(name===row.predicted)tr.className='selected';for(const value of [i,LABELS[i],fmt(row.logits[i]),pct(row.probabilities[i])]){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);});return table;
  }
  function addPreview(parent,title,src){const box=document.createElement('div');box.className='preview-block';const label=document.createElement('div');label.className='preview-title';label.textContent=title;const img=document.createElement('img');img.src=src;img.alt=title;box.append(label,img);parent.append(box);}
  function downloadTensor(row){
    const payload={file:row.name,model_outputs:{roi_segmenter:{name:'mask_logits',shape:[1,1,256,256],statistics:row.maskStats,threshold:0.45,active_pixels:row.box.activePixels,raw_values:row.maskLogits},roi_classifier:{name:'class_logits',shape:[1,9],classes:CLASSES.map((name,i)=>({id:i,name,logit:row.logits[i],softmax:row.probabilities[i]}))}},roi_xywh_original:[row.box.x,row.box.y,row.box.w,row.box.h],inference_ms:row.ms};
    const blob=new Blob([JSON.stringify(payload)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${row.file.name.replace(/\.[^.]+$/,'')}_onnx_output.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function drawResults(){
    $('resultCount').textContent=`${state.rows.length} ảnh đã chạy`;
    for(const url of state.urls)URL.revokeObjectURL(url);state.urls=[];
    const container=$('results');container.replaceChildren();
    for(const row of state.rows){
      const card=document.createElement('article');card.className='item';if(row.trueLabel)card.classList.add(row.trueLabel===row.predicted?'correct':'wrong');
      if(!row.error){const previews=document.createElement('div');previews.className='previews';addPreview(previews,'Ảnh + mask ROI',row.overlay);addPreview(previews,'ROI đưa vào classifier',row.roiImage);card.append(previews);}
      const details=document.createElement('div');details.className='details';
      const filename=document.createElement('div');filename.className='filename';filename.title=row.name;filename.textContent=row.name;
      const pred=document.createElement('div');pred.className='pred';pred.textContent=row.error?'Lỗi đọc ảnh':`${LABELS[CLASSES.indexOf(row.predicted)]} · ${pct(row.confidence)}`;
      details.append(filename,pred);
      if(row.trueLabel){const badge=document.createElement('span');badge.className='badge';badge.textContent=`Nhãn thật: ${LABELS[CLASSES.indexOf(row.trueLabel)]}`;details.append(badge);}
      if(row.error){const error=document.createElement('div');error.className='sub';error.textContent=row.error;details.append(error);}else{
        const stats=document.createElement('div');stats.className='tensor-stats';
        stats.innerHTML=`<b>Thời gian:</b> ${row.ms.toFixed(2)} ms<br><b>Mask output:</b> [1,1,256,256] · logit min ${fmt(row.maskStats.min)} · max ${fmt(row.maskStats.max)} · mean ${fmt(row.maskStats.mean)}<br><b>Pixel mask ≥ 0.45:</b> ${row.box.activePixels}/65,536<br><b>ROI ảnh gốc (x,y,w,h):</b> ${row.box.x.toFixed(1)}, ${row.box.y.toFixed(1)}, ${row.box.w.toFixed(1)}, ${row.box.h.toFixed(1)}${row.box.fallback?' · fallback toàn ảnh':''}<br><b>Classifier output:</b> [1,9] logits + softmax`;
        details.append(stats,renderScores(row));const dl=document.createElement('button');dl.className='download-output';dl.textContent='Tải toàn bộ tensor output JSON';dl.addEventListener('click',()=>downloadTensor(row));details.append(dl);
      }
      card.append(details);container.append(card);
    }
    showSummary();drawMatrix();$('exportButton').disabled=state.rows.length===0;
  }
  function metricCard(parent,title,value){const item=document.createElement('div');item.className='metric';const a=document.createElement('span');a.textContent=title;const b=document.createElement('strong');b.textContent=value;item.append(a,b);parent.append(item);}
  function showSummary(){
    const valid=state.rows.filter(r=>!r.error),labelled=valid.filter(r=>r.trueLabel),times=valid.map(r=>r.ms).sort((a,b)=>a-b),conf=valid.map(r=>r.confidence);
    const accuracy=labelled.length?labelled.filter(r=>r.trueLabel===r.predicted).length/labelled.length:null;
    const total=times.reduce((a,b)=>a+b,0),mean=times.length?total/times.length:0,median=times.length?times[Math.floor(times.length/2)]:0;
    const summary=$('summary');summary.replaceChildren();summary.hidden=valid.length===0;
    [['Ảnh thành công',`${valid.length}/${state.rows.length}`],['Có nhãn thật',String(labelled.length)],['Accuracy',accuracy===null?'—':pct(accuracy)],['Tải model',`${state.modelLoadMs.toFixed(0)} ms`],['Tổng inference',`${total.toFixed(1)} ms`],['TB / ảnh',`${mean.toFixed(2)} ms`],['Median / ảnh',`${median.toFixed(2)} ms`],['Min / max',times.length?`${times[0].toFixed(2)} / ${times[times.length-1].toFixed(2)} ms`:'—'],['Confidence TB',conf.length?pct(conf.reduce((a,b)=>a+b,0)/conf.length):'—'],['Ảnh lỗi',String(state.rows.length-valid.length)]].forEach(([a,b])=>metricCard(summary,a,b));
  }
  function getMetrics(rows){
    const matrix=CLASSES.map(()=>CLASSES.map(()=>0));
    for(const r of rows){const a=CLASSES.indexOf(r.trueLabel),b=CLASSES.indexOf(r.predicted);if(a>=0&&b>=0)matrix[a][b]++;}
    const reports=CLASSES.map((name,i)=>{const tp=matrix[i][i],support=matrix[i].reduce((a,b)=>a+b,0),predicted=matrix.reduce((a,r)=>a+r[i],0),precision=predicted?tp/predicted:0,recall=support?tp/support:0,f1=precision+recall?2*precision*recall/(precision+recall):0;return{name,support,tp,precision,recall,f1};});
    const accuracy=rows.length?reports.reduce((s,r)=>s+r.tp,0)/rows.length:0;
    const macro=x=>reports.filter(r=>r.support>0).reduce((s,r)=>s+r[x],0)/(reports.filter(r=>r.support>0).length||1);
    return{matrix,reports,accuracy,macroPrecision:macro('precision'),macroRecall:macro('recall'),macroF1:macro('f1')};
  }
  function drawMatrix(){
    const rows=state.rows.filter(r=>r.trueLabel&&!r.error),card=$('matrixCard'),reportCard=$('classMetricsCard');
    card.hidden=reportCard.hidden=rows.length===0;if(!rows.length)return;
    const m=getMetrics(rows);$('matrixAccuracy').textContent=`N=${rows.length} · accuracy ${pct(m.accuracy)} · macro P/R/F1 ${pct(m.macroPrecision)} / ${pct(m.macroRecall)} / ${pct(m.macroF1)}`;
    const table=$('matrixTable');table.replaceChildren();const head=document.createElement('tr');for(const name of ['Thật ↓ / Đoán →',...LABELS]){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
    for(let i=0;i<CLASSES.length;i++){const tr=document.createElement('tr'),th=document.createElement('th');th.textContent=LABELS[i];tr.append(th);const support=m.reports[i].support;for(let j=0;j<CLASSES.length;j++){const td=document.createElement('td'),n=m.matrix[i][j];td.innerHTML=`${n}<small>${support?pct(n/support):'—'}</small>`;if(i===j)td.className='diag';else if(n)td.className='off';tr.append(td);}table.append(tr);}
    const ct=$('classMetricsTable');ct.replaceChildren();const hr=document.createElement('tr');for(const x of ['Lớp','Support','Đúng','Precision','Recall','F1']){const th=document.createElement('th');th.textContent=x;hr.append(th);}ct.append(hr);
    for(const r of m.reports){const tr=document.createElement('tr');for(const x of [LABELS[CLASSES.indexOf(r.name)],r.support,r.tp,pct(r.precision),pct(r.recall),pct(r.f1)]){const td=document.createElement('td');td.textContent=x;tr.append(td);}ct.append(tr);}
  }
  async function run(){
    if(state.busy||!state.files.length||!state.segmenter)return;
    state.busy=true;$('runButton').disabled=true;$('exportButton').disabled=true;state.rows=[];$('results').replaceChildren();$('summary').hidden=$('matrixCard').hidden=$('classMetricsCard').hidden=true;
    const files=state.files;let failed=0;
    try{for(let i=0;i<files.length;i++){
      status(`Đang chạy ${i+1}/${files.length}: ${files[i].name}`);
      try{state.rows.push(await predict(files[i]));}catch(error){failed++;state.rows.push({file:files[i],name:files[i].webkitRelativePath||files[i].name,error:error.message||String(error),ms:0,box:{fallback:true}});}
      $('progressBar').style.width=`${100*(i+1)/files.length}%`;if(i%3===0||i===files.length-1)drawResults();await new Promise(resolve=>setTimeout(resolve,0));
    }drawResults();status(`Xong ${files.length} ảnh · lỗi ${failed}.`);}finally{state.busy=false;$('runButton').disabled=!state.segmenter||state.files.length===0;}
  }
  function exportCsv(){
    const cols=['file','true_label','predicted_label','confidence','inference_ms','roi_x','roi_y','roi_width','roi_height','mask_min','mask_max','mask_mean','mask_active_pixels','error'],quote=x=>`"${String(x??'').replaceAll('"','""')}"`,lines=[cols.map(quote).join(',')];
    for(const r of state.rows)lines.push([r.name,r.trueLabel||'',r.predicted||'',r.confidence??'',r.ms?.toFixed(3)||'',r.box?.x??'',r.box?.y??'',r.box?.w??'',r.box?.h??'',r.maskStats?.min??'',r.maskStats?.max??'',r.maskStats?.mean??'',r.box?.activePixels??'',r.error||''].map(quote).join(','));
    const blob=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='roi_predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  $('singleInput').addEventListener('change',e=>setFiles(e.target.files));$('folderInput').addEventListener('change',e=>setFiles(e.target.files));$('runButton').addEventListener('click',run);$('exportButton').addEventListener('click',exportCsv);init();
})();
