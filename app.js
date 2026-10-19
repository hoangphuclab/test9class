(() => {
  'use strict';
  const CLASSES = ['bangtotnghiep','cccd','dtut','gxnkqthi','hocba10','hocba11','hocba12','hocbabia','vsat'];
  const LABELS = ['Bằng tốt nghiệp','CCCD','ĐTUT','GXN kết quả thi','Học bạ 10','Học bạ 11','Học bạ 12','Học bạ bìa','VSAT'];
  const MEAN = [0.485,0.456,0.406], STD = [0.229,0.224,0.225];
  const $ = id => document.getElementById(id);
  const state = {files:[], segmenter:null, classifier:null, prototypes:null, rows:[], urls:[], busy:false, modelLoadMs:0, resultsFilter:'all',cameraStream:null};
  const status = message => { $('runStatus').textContent=message; };

  async function init(){
    const started=performance.now();
    const loadStatus=message=>{$('modelStatus').textContent=message;status(message);};
    try{
      if(!window.ort) throw new Error('Không tải được ONNX Runtime.');
      ort.env.wasm.wasmPaths={mjs:new URL('vendor/ort-wasm-simd-threaded.mjs',document.baseURI).href,wasm:new URL('vendor/ort-wasm-simd-threaded.wasm',document.baseURI).href};
      ort.env.wasm.numThreads=1; ort.env.wasm.proxy=false;
      loadStatus('Đang khởi tạo WebAssembly…');
      const loadSession=async(path,name)=>{
        loadStatus(`Đang tải ${name}…`);
        const response=await fetch(path);
        if(!response.ok)throw new Error(`Không tải được ${path} (${response.status})`);
        const modelBytes=await response.arrayBuffer();
        loadStatus(`Đang khởi tạo ${name} (${(modelBytes.byteLength/1048576).toFixed(1)} MB)…`);
        return ort.InferenceSession.create(new Uint8Array(modelBytes),{executionProviders:['wasm'],graphOptimizationLevel:'basic'});
      };
      state.segmenter=await loadSession('models/roi_segmenter.onnx','model tìm ROI');
      state.classifier=await loadSession('models/roi_classifier.onnx','model phân loại + embedding');
      loadStatus('Đang nạp vector prototype…');
      const prototypeResponse=await fetch('models/prototypes.json');
      if(!prototypeResponse.ok)throw new Error(`Không tải được models/prototypes.json (${prototypeResponse.status})`);
      state.prototypes=await prototypeResponse.json();
      $('confidenceThreshold').value=state.prototypes.suggested_thresholds?.confidence??0.39;
      $('cosineThreshold').value=state.prototypes.suggested_thresholds?.cosine??0.39;
      updateThresholdLabels();
      $('thresholdNote').textContent='Mặc định confidence và cosine đều 0.39. Từ chối nếu một trong hai thấp hơn ngưỡng; hai ngưỡng có thể kéo chỉnh.';
      state.modelLoadMs=performance.now()-started;
      $('modelStatus').textContent=`Sẵn sàng · tải ${state.modelLoadMs.toFixed(0)} ms`;
      $('runButton').disabled=state.files.length===0;
      $('captureButton').disabled=!state.cameraStream;
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

  function resizeCameraStage(){
    const video=$('cameraVideo'),stage=$('cameraStage');
    if(!video.videoWidth||!video.videoHeight)return;
    const ratio=video.videoWidth/video.videoHeight,availableWidth=stage.parentElement.clientWidth;
    stage.style.aspectRatio=`${video.videoWidth}/${video.videoHeight}`;
    stage.style.width=`${Math.min(availableWidth,window.innerHeight*0.68*ratio,1100)}px`;
  }
  async function startCamera(){
    if(!navigator.mediaDevices?.getUserMedia){$('cameraStatus').textContent='Camera cần HTTPS hoặc localhost';return;}
    $('startCameraButton').disabled=true;$('cameraStatus').textContent='Đang mở camera…';
    try{
      state.cameraStream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}}});
      const video=$('cameraVideo');video.srcObject=state.cameraStream;
      await new Promise((resolve,reject)=>{if(video.readyState>=1&&video.videoWidth)resolve();else{video.onloadedmetadata=resolve;video.onerror=()=>reject(new Error('Không đọc được luồng camera.'));}});
      await video.play();resizeCameraStage();$('cameraStage').hidden=false;$('cameraPlaceholder').hidden=true;
      $('captureButton').disabled=!state.segmenter;$('stopCameraButton').disabled=false;
      $('cameraStatus').textContent=`Camera sẵn sàng · ${video.videoWidth} × ${video.videoHeight}`;
    }catch(error){stopCamera();$('cameraStatus').textContent=error.name==='NotAllowedError'?'Chưa được cấp quyền camera':(error.message||'Không mở được camera');}
    finally{$('startCameraButton').disabled=false;}
  }
  function stopCamera(){
    if(state.cameraStream)for(const track of state.cameraStream.getTracks())track.stop();
    state.cameraStream=null;$('cameraVideo').srcObject=null;$('cameraStage').hidden=true;$('cameraPlaceholder').hidden=false;
    $('captureButton').disabled=true;$('stopCameraButton').disabled=true;$('cameraStatus').textContent='Camera chưa bật';
  }
  async function captureAndPredict(){
    if(!state.cameraStream||state.busy)return;
    const video=$('cameraVideo'),canvas=document.createElement('canvas');
    const frameHeight=0.90,frameWidth=frameHeight*(210/297),sourceWidth=video.videoWidth*frameWidth,sourceHeight=video.videoHeight*frameHeight;
    canvas.width=Math.round(sourceWidth);canvas.height=Math.round(sourceHeight);
    const sx=(video.videoWidth-sourceWidth)/2,sy=(video.videoHeight-sourceHeight)/2;
    canvas.getContext('2d').drawImage(video,sx,sy,sourceWidth,sourceHeight,0,0,canvas.width,canvas.height);
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',0.94));
    if(!blob){$('cameraStatus').textContent='Không chụp được ảnh';return;}
    const file=new File([blob],`camera_${new Date().toISOString().replace(/[:.]/g,'-')}.jpg`,{type:'image/jpeg'});
    setFiles([file]);$('singleInput').value='';$('folderInput').value='';
    await run();
  }

  const loadBitmap=file=>createImageBitmap(file,{imageOrientation:'from-image'});
  function letterboxCanvas(bitmap,size,sourceBox=null){
    const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.fillStyle='#fff';ctx.fillRect(0,0,size,size);
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
  function grayscaleCanvas(source){
    const canvas=document.createElement('canvas');canvas.width=source.width;canvas.height=source.height;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(source,0,0);
    const image=ctx.getImageData(0,0,canvas.width,canvas.height),data=image.data;
    for(let i=0;i<data.length;i+=4){const value=Math.round((299*data[i]+587*data[i+1]+114*data[i+2])/1000);data[i]=value;data[i+1]=value;data[i+2]=value;}
    ctx.putImageData(image,0,0);return canvas;
  }
  function findBox(maskData,bitmap,lb){
    const size=256*256,prob=new Float32Array(size),seen=new Uint8Array(size),queue=new Int32Array(size);
    let count=0;
    for(let i=0;i<size;i++){const p=1/(1+Math.exp(-maskData[i]));prob[i]=p;if(p>=0.45)count++;}
    let best=null;
    for(let start=0;start<size;start++){
      if(seen[start]||prob[start]<0.45)continue;
      let head=0,tail=0,sum=0,minX=256,minY=256,maxX=-1,maxY=-1;
      queue[tail++]=start;seen[start]=1;
      while(head<tail){
        const idx=queue[head++],x=idx%256,y=(idx/256)|0;
        sum+=prob[idx];minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
          if(dx===0&&dy===0)continue;const nx=x+dx,ny=y+dy;
          if(nx<0||ny<0||nx>=256||ny>=256)continue;const ni=ny*256+nx;
          if(!seen[ni]&&prob[ni]>=0.45){seen[ni]=1;queue[tail++]=ni;}
        }
      }
      if(tail<12)continue;
      const meanConfidence=sum/tail;
      if(!best||meanConfidence>best.meanConfidence||(meanConfidence===best.meanConfidence&&tail>best.componentPixels))
        best={minX,minY,maxX,maxY,componentPixels:tail,meanConfidence,componentIndices:queue.slice(0,tail)};
    }
    if(!best)return{x:0,y:0,w:bitmap.width,h:bitmap.height,fallback:true,maskBox:[0,0,256,256],activePixels:count,componentPixels:0,meanConfidence:0,componentMask:new Uint8Array(size)};
    const {minX,minY,maxX,maxY}=best;
    let x0=minX,y0=minY,x1=maxX+1,y1=maxY+1;
    const mx=Math.max(2,Math.round((x1-x0)*0.04)),my=Math.max(2,Math.round((y1-y0)*0.08));
    x0=Math.max(0,x0-mx);y0=Math.max(0,y0-my);x1=Math.min(256,x1+mx);y1=Math.min(256,y1+my);
    const {scale,left,top}=lb,sx=Math.max(0,Math.min(bitmap.width,(x0-left)/scale)),sy=Math.max(0,Math.min(bitmap.height,(y0-top)/scale));
    const ex=Math.max(sx,Math.min(bitmap.width,(x1-left)/scale)),ey=Math.max(sy,Math.min(bitmap.height,(y1-top)/scale));
    const componentMask=new Uint8Array(size);for(const index of best.componentIndices)componentMask[index]=1;
    return{x:sx,y:sy,w:Math.max(1,ex-sx),h:Math.max(1,ey-sy),fallback:false,maskBox:[x0,y0,x1,y1],activePixels:count,componentPixels:best.componentPixels,meanConfidence:best.meanConfidence,componentMask};
  }
  function softmax(logits){let max=-Infinity;for(const x of logits)max=Math.max(max,x);const e=Array.from(logits,x=>Math.exp(x-max)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);}
  function prototypeCosines(embedding){
    const norm=Math.sqrt(embedding.reduce((sum,x)=>sum+x*x,0))||1;
    const normalized=embedding.map(x=>x/norm);
    return state.prototypes.embeddings.map((prototype,i)=>({label:state.prototypes.classes[i],cosine:prototype.reduce((sum,x,j)=>sum+x*normalized[j],0)}));
  }
  function currentThresholds(){return{confidence:Number($('confidenceThreshold').value),cosine:Number($('cosineThreshold').value)};}
  function applyDecision(row){
    if(row.error)return row;
    const thresholds=currentThresholds();
    row.accepted=row.confidence>=thresholds.confidence&&row.nearestPrototype.cosine>=thresholds.cosine;
    row.rejectionReasons=[];
    if(row.confidence<thresholds.confidence)row.rejectionReasons.push(`confidence ${pct(row.confidence)} < ${pct(thresholds.confidence)}`);
    if(row.nearestPrototype.cosine<thresholds.cosine)row.rejectionReasons.push(`cosine ${row.nearestPrototype.cosine.toFixed(4)} < ${thresholds.cosine.toFixed(2)}`);
    return row;
  }
  function updateThresholdLabels(){
    $('confidenceValue').textContent=pct(Number($('confidenceThreshold').value));
    $('cosineValue').textContent=Number($('cosineThreshold').value).toFixed(2);
  }
  function tensorStats(values){let min=Infinity,max=-Infinity,sum=0;for(const value of values){min=Math.min(min,value);max=Math.max(max,value);sum+=value;}return{count:values.length,min,max,mean:sum/values.length};}
  function overlayUrl(pageCanvas,maskData,box){
    const canvas=document.createElement('canvas');canvas.width=canvas.height=256;const ctx=canvas.getContext('2d');ctx.drawImage(pageCanvas,0,0);
    const image=ctx.getImageData(0,0,256,256);
    for(let i=0;i<maskData.length;i++){if(box.componentMask?.[i]){const k=i*4;image.data[k]=Math.round(image.data[k]*0.38+255*0.62);image.data[k+1]=Math.round(image.data[k+1]*0.38+42*0.62);image.data[k+2]=Math.round(image.data[k+2]*0.38+42*0.62);}}
    ctx.putImageData(image,0,0);
    const [x0,y0,x1,y1]=box.maskBox;ctx.strokeStyle='#00e676';ctx.lineWidth=2;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    return canvas.toDataURL('image/png');
  }
  function maskUrl(maskData,box){
    const canvas=document.createElement('canvas');canvas.width=canvas.height=256;
    const ctx=canvas.getContext('2d'),image=ctx.createImageData(256,256);
    for(let i=0;i<maskData.length;i++){
      const p=1/(1+Math.exp(-maskData[i])),v=box.componentMask?.[i]?Math.round(p*255):0,k=i*4;
      image.data[k]=v;image.data[k+1]=v;image.data[k+2]=v;image.data[k+3]=255;
    }
    ctx.putImageData(image,0,0);return canvas.toDataURL('image/png');
  }
  async function predict(file){
    const started=performance.now(),bitmap=await loadBitmap(file);
    try{
      const trueLabel=labelFromPath(file.webkitRelativePath||file.name);
      const processedBitmap=grayscaleCanvas(bitmap),page=letterboxCanvas(processedBitmap,256);
      const maskOut=await state.segmenter.run({page:tensorFromCanvas(page.canvas)});
      const mask=maskOut.mask_logits.data,box=findBox(mask,bitmap,page);
      const crop=letterboxCanvas(processedBitmap,224,[box.x,box.y,box.w,box.h]);
      const classifierOut=await state.classifier.run({roi:tensorFromCanvas(crop.canvas)});
      const logits=Array.from(classifierOut.class_logits.data),probabilities=softmax(logits);
      const predicted=probabilities.indexOf(Math.max(...probabilities));
      const embedding=Array.from(classifierOut.embedding.data),cosines=prototypeCosines(embedding);
      const nearestPrototype=cosines.reduce((a,b)=>b.cosine>a.cosine?b:a,cosines[0]);
      const maskStats=tensorStats(mask);
      return{file,name:file.webkitRelativePath||file.name,trueLabel,predicted:CLASSES[predicted],confidence:probabilities[predicted],probabilities,logits,embedding,cosines,nearestPrototype,grayscaleApplied:true,maskLogits:Array.from(mask),maskStats,box,ms:performance.now()-started,grayscaleImage:processedBitmap.toDataURL('image/jpeg',0.92),overlay:overlayUrl(page.canvas,mask,box),maskImage:maskUrl(mask,box),roiImage:crop.canvas.toDataURL('image/jpeg',0.92),error:''};
    }finally{bitmap.close();}
  }
  function labelFromPath(path){const parts=path.replaceAll('\\','/').split('/').map(x=>x.toLowerCase());for(let i=parts.length-1;i>=0;i--){const found=CLASSES.indexOf(parts[i]);if(found>=0)return CLASSES[found];}return'';}
  const pct=x=>`${(x*100).toFixed(2)}%`,fmt=x=>Number(x).toFixed(5);

  function renderScores(row){
    const table=document.createElement('table');table.className='score-table';
    const head=document.createElement('tr');for(const name of ['ID','Lớp','Logit','Softmax','Cosine prototype']){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
    CLASSES.forEach((name,i)=>{const tr=document.createElement('tr');if(name===row.predicted)tr.className='selected';const cosine=row.cosines.find(item=>item.label===name)?.cosine??0;for(const value of [i,LABELS[i],fmt(row.logits[i]),pct(row.probabilities[i]),cosine.toFixed(4)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);});return table;
  }
  function addPreview(parent,title,src){const box=document.createElement('div');box.className='preview-block';const label=document.createElement('div');label.className='preview-title';label.textContent=title;const img=document.createElement('img');img.src=src;img.alt=title;box.append(label,img);parent.append(box);}
  function modelOutputPayload(row){
    applyDecision(row);
    return{file:row.name,model_outputs:{roi_segmenter:{name:'mask_logits',shape:[1,1,256,256],statistics:row.maskStats,threshold:0.45,selection:'one 8-connected component with the highest mean sigmoid confidence among components of at least 12 pixels',active_pixels:row.box.activePixels,selected_component_pixels:row.box.componentPixels,selected_component_mean_confidence:row.box.meanConfidence,raw_values:row.maskLogits},roi_classifier:{name:'class_logits',shape:[1,9],classes:CLASSES.map((name,i)=>({id:i,name,logit:row.logits[i],softmax:row.probabilities[i]})),embedding:{name:'embedding',shape:[1,row.embedding.length],dimensions:row.embedding.length,raw_values:row.embedding},prototype_cosine:{metric:'cosine',scores:row.cosines,nearest:row.nearestPrototype}}},rejection:{accepted:row.accepted,reasons:row.rejectionReasons,thresholds:currentThresholds()},preprocessing:{grayscale_applied_to_all_images:row.grayscaleApplied,true_label:row.trueLabel||null},roi_xywh_original:[row.box.x,row.box.y,row.box.w,row.box.h],inference_ms:row.ms};
  }
  function downloadTensor(row){
    const payload=modelOutputPayload(row);
    const blob=new Blob([JSON.stringify(payload)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${row.file.name.replace(/\.[^.]+$/,'')}_onnx_output.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function drawResults(){
    state.rows.forEach(applyDecision);
    const labelled=state.rows.filter(row=>row.trueLabel&&!row.error);
    const wrong=labelled.filter(row=>row.predicted!==row.trueLabel).length;
    const selected=state.rows.filter(row=>{
      if(state.resultsFilter==='wrong')return row.trueLabel&&!row.error&&row.predicted!==row.trueLabel;
      if(state.resultsFilter==='correct')return row.trueLabel&&!row.error&&row.predicted===row.trueLabel&&row.accepted;
      if(state.resultsFilter==='rejected')return !row.error&&!row.accepted;
      return true;
    });
    $('resultCount').textContent=`Hiện ${selected.length}/${state.rows.length} ảnh · sai ${wrong}`;
    for(const url of state.urls)URL.revokeObjectURL(url);state.urls=[];
    const container=$('results');container.replaceChildren();
    if(!selected.length){const empty=document.createElement('div');empty.className='empty';empty.textContent=state.rows.length?'Không có ảnh phù hợp với bộ lọc.':'Chọn ảnh hoặc thư mục.';container.append(empty);}
    for(const row of selected){
      applyDecision(row);
      const card=document.createElement('article');card.className='item';if(!row.accepted&&!row.error)card.classList.add('refused');else if(row.trueLabel)card.classList.add(row.trueLabel===row.predicted?'correct':'wrong');
      if(!row.error){const previews=document.createElement('div');previews.className='previews';addPreview(previews,'Ảnh grayscale đưa vào model',row.grayscaleImage);addPreview(previews,'Ảnh + mask ROI',row.overlay);addPreview(previews,'Mask xác suất',row.maskImage);addPreview(previews,'ROI grayscale đưa vào classifier',row.roiImage);card.append(previews);}
      const details=document.createElement('div');details.className='details';
      const filename=document.createElement('div');filename.className='filename';filename.title=row.name;filename.textContent=row.name;
      const pred=document.createElement('div');pred.className='pred';pred.textContent=row.error?'Lỗi đọc ảnh':row.accepted?`${LABELS[CLASSES.indexOf(row.predicted)]} · ${pct(row.confidence)}`:`KHÔNG PHẢI GIẤY TỜ hoặc ẢNH MỜ · ${row.rejectionReasons.join(' · ')}`;
      details.append(filename,pred);
      if(row.trueLabel){const badge=document.createElement('span');badge.className='badge';badge.textContent=`Nhãn thật: ${LABELS[CLASSES.indexOf(row.trueLabel)]}`;details.append(badge);}
      if(row.error){const error=document.createElement('div');error.className='sub';error.textContent=row.error;details.append(error);}else{
        const stats=document.createElement('div');stats.className='tensor-stats';
        stats.innerHTML=`<b>Ứng viên:</b> ${LABELS[CLASSES.indexOf(row.predicted)]} · confidence ${pct(row.confidence)}<br><b>Ngưỡng:</b> confidence ${pct(currentThresholds().confidence)} · cosine ${currentThresholds().cosine.toFixed(2)}<br><b>Mask output:</b> [1,1,256,256] · logit min ${fmt(row.maskStats.min)} · max ${fmt(row.maskStats.max)} · mean ${fmt(row.maskStats.mean)}<br><b>Mask ≥ 0.45:</b> ${row.box.activePixels} pixel · chọn 1 vùng liên thông confidence TB cao nhất: ${row.box.componentPixels} pixel, confidence TB ${pct(row.box.meanConfidence)}<br><b>ROI ảnh gốc (x,y,w,h):</b> ${row.box.x.toFixed(1)}, ${row.box.y.toFixed(1)}, ${row.box.w.toFixed(1)}, ${row.box.h.toFixed(1)}${row.box.fallback?' · không có vùng đạt ngưỡng, fallback toàn ảnh':''}<br><b>Đầu vào:</b> grayscale · áp dụng cho cả 9 nhãn<br><b>Classifier output:</b> [1,9] logits + softmax<br><b>Embedding:</b> ${row.embedding.length} chiều · cosine gần nhất ${row.nearestPrototype.label}: ${row.nearestPrototype.cosine.toFixed(4)}<br><b>Quyết định:</b> ${row.accepted?'chấp nhận là một trong 9 lớp':'từ chối theo ngưỡng'}`;
        details.append(stats,renderScores(row));const raw=document.createElement('details');raw.className='raw-output';const rawTitle=document.createElement('summary');rawTitle.textContent='Toàn bộ raw model output';const rawPre=document.createElement('pre');raw.append(rawTitle,rawPre);raw.addEventListener('toggle',()=>{if(raw.open&&!rawPre.textContent)rawPre.textContent=JSON.stringify(modelOutputPayload(row),null,2);});details.append(raw);const dl=document.createElement('button');dl.className='download-output';dl.textContent='Tải toàn bộ tensor output JSON';dl.addEventListener('click',()=>downloadTensor(row));details.append(dl);
      }
      card.append(details);container.append(card);
    }
    showSummary();drawMatrix();$('exportButton').disabled=state.rows.length===0;
  }
  function metricCard(parent,title,value){const item=document.createElement('div');item.className='metric';const a=document.createElement('span');a.textContent=title;const b=document.createElement('strong');b.textContent=value;item.append(a,b);parent.append(item);}
  function showSummary(){
    const valid=state.rows.filter(r=>!r.error),labelled=valid.filter(r=>r.trueLabel),times=valid.map(r=>r.ms).sort((a,b)=>a-b),conf=valid.map(r=>r.confidence);
    const accuracy=labelled.length?labelled.filter(r=>r.accepted&&r.trueLabel===r.predicted).length/labelled.length:null;
    const total=times.reduce((a,b)=>a+b,0),mean=times.length?total/times.length:0,median=times.length?times[Math.floor(times.length/2)]:0;
    const summary=$('summary');summary.replaceChildren();summary.hidden=valid.length===0;
    [['Ảnh thành công',`${valid.length}/${state.rows.length}`],['Được nhận',String(valid.filter(r=>r.accepted).length)],['Bị từ chối',String(valid.filter(r=>!r.accepted).length)],['Có nhãn thật',String(labelled.length)],['Accuracy',accuracy===null?'—':pct(accuracy)],['Tải model',`${state.modelLoadMs.toFixed(0)} ms`],['Tổng inference',`${total.toFixed(1)} ms`],['TB / ảnh',`${mean.toFixed(2)} ms`],['Median / ảnh',`${median.toFixed(2)} ms`],['Confidence TB',conf.length?pct(conf.reduce((a,b)=>a+b,0)/conf.length):'—'],['Ảnh lỗi',String(state.rows.length-valid.length)]].forEach(([a,b])=>metricCard(summary,a,b));
  }
  function getMetrics(rows){
    const matrix=CLASSES.map(()=>Array(CLASSES.length+1).fill(0));
    for(const r of rows){const a=CLASSES.indexOf(r.trueLabel),b=r.accepted?CLASSES.indexOf(r.predicted):CLASSES.length;if(a>=0)matrix[a][b>=0?b:CLASSES.length]++;}
    const reports=CLASSES.map((name,i)=>{const tp=matrix[i][i],support=matrix[i].reduce((a,b)=>a+b,0),predicted=matrix.reduce((a,r)=>a+r[i],0),precision=predicted?tp/predicted:0,recall=support?tp/support:0,f1=precision+recall?2*precision*recall/(precision+recall):0;return{name,support,tp,precision,recall,f1};});
    const accuracy=rows.length?reports.reduce((s,r)=>s+r.tp,0)/rows.length:0;
    const macro=x=>reports.filter(r=>r.support>0).reduce((s,r)=>s+r[x],0)/(reports.filter(r=>r.support>0).length||1);
    return{matrix,reports,accuracy,macroPrecision:macro('precision'),macroRecall:macro('recall'),macroF1:macro('f1')};
  }
  function drawMatrix(){
    const rows=state.rows.filter(r=>r.trueLabel&&!r.error),card=$('matrixCard'),reportCard=$('classMetricsCard');
    card.hidden=reportCard.hidden=rows.length===0;if(!rows.length)return;
    const m=getMetrics(rows);$('matrixAccuracy').textContent=`N=${rows.length} · accuracy ${pct(m.accuracy)} · macro P/R/F1 ${pct(m.macroPrecision)} / ${pct(m.macroRecall)} / ${pct(m.macroF1)}`;
    const table=$('matrixTable');table.replaceChildren();const head=document.createElement('tr');for(const name of ['Thật ↓ / Đoán →',...LABELS,'Từ chối']){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
    for(let i=0;i<CLASSES.length;i++){const tr=document.createElement('tr'),th=document.createElement('th');th.textContent=LABELS[i];tr.append(th);const support=m.reports[i].support;for(let j=0;j<CLASSES.length+1;j++){const td=document.createElement('td'),n=m.matrix[i][j];td.innerHTML=`${n}<small>${support?pct(n/support):'—'}</small>`;if(i===j)td.className='diag';else if(n)td.className='off';tr.append(td);}table.append(tr);}
    const ct=$('classMetricsTable');ct.replaceChildren();const hr=document.createElement('tr');for(const x of ['Lớp','Support','Đúng','Precision','Recall','F1']){const th=document.createElement('th');th.textContent=x;hr.append(th);}ct.append(hr);
    for(const r of m.reports){const tr=document.createElement('tr');for(const x of [LABELS[CLASSES.indexOf(r.name)],r.support,r.tp,pct(r.precision),pct(r.recall),pct(r.f1)]){const td=document.createElement('td');td.textContent=x;tr.append(td);}ct.append(tr);}
  }
  async function run(){
    if(state.busy||!state.files.length||!state.segmenter)return;
    state.busy=true;$('runButton').disabled=true;$('captureButton').disabled=true;$('exportButton').disabled=true;state.rows=[];$('results').replaceChildren();$('summary').hidden=$('matrixCard').hidden=$('classMetricsCard').hidden=true;
    const files=state.files;let failed=0;
    try{for(let i=0;i<files.length;i++){
      status(`Đang chạy ${i+1}/${files.length}: ${files[i].name}`);
      try{state.rows.push(await predict(files[i]));}catch(error){failed++;state.rows.push({file:files[i],name:files[i].webkitRelativePath||files[i].name,error:error.message||String(error),ms:0,box:{fallback:true}});}
      $('progressBar').style.width=`${100*(i+1)/files.length}%`;if(i%3===0||i===files.length-1)drawResults();await new Promise(resolve=>setTimeout(resolve,0));
    }drawResults();status(`Xong ${files.length} ảnh · lỗi ${failed}.`);}finally{state.busy=false;$('runButton').disabled=!state.segmenter||state.files.length===0;$('captureButton').disabled=!state.cameraStream||!state.segmenter;}
  }
  function exportCsv(){
    const cols=['file','true_label','candidate_label','accepted_as_document','rejection_reason','confidence','nearest_prototype','prototype_cosine','inference_ms','roi_x','roi_y','roi_width','roi_height','mask_min','mask_max','mask_mean','mask_active_pixels','error'],quote=x=>`"${String(x??'').replaceAll('"','""')}"`,lines=[cols.map(quote).join(',')];
    for(const r of state.rows){applyDecision(r);lines.push([r.name,r.trueLabel||'',r.predicted||'',r.accepted??'',r.rejectionReasons?.join('; ')||'',r.confidence??'',r.nearestPrototype?.label||'',r.nearestPrototype?.cosine??'',r.ms?.toFixed(3)||'',r.box?.x??'',r.box?.y??'',r.box?.w??'',r.box?.h??'',r.maskStats?.min??'',r.maskStats?.max??'',r.maskStats?.mean??'',r.box?.activePixels??'',r.error||''].map(quote).join(','));}
    const blob=new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='roi_predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  const thresholdsChanged=()=>{updateThresholdLabels();if(state.rows.length)drawResults();};
  $('resultFilter').addEventListener('change',event=>{state.resultsFilter=event.target.value;drawResults();});
  $('confidenceThreshold').addEventListener('input',thresholdsChanged);$('cosineThreshold').addEventListener('input',thresholdsChanged);
  $('singleInput').addEventListener('change',e=>setFiles(e.target.files));$('folderInput').addEventListener('change',e=>setFiles(e.target.files));$('runButton').addEventListener('click',run);$('exportButton').addEventListener('click',exportCsv);
  $('startCameraButton').addEventListener('click',startCamera);$('captureButton').addEventListener('click',captureAndPredict);$('stopCameraButton').addEventListener('click',stopCamera);
  window.addEventListener('resize',resizeCameraStage);window.addEventListener('pagehide',stopCamera);init();
})();
