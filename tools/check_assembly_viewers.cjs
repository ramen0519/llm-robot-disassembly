// Visual smoke checks on newly generated local previews, using bundled Chromium.
const {chromium} = require(process.argv[2]);
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
(async()=>{
  const input = path.resolve(process.argv[3]);
  const output = path.resolve(process.argv[4] || process.argv[3]);
  if(output!==input)fs.mkdirSync(output);
  const browser = await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const result=[];
  try {
    for(const id of ['07009','02628']){
      const page=await browser.newPage({viewport:{width:800,height:1100},deviceScaleFactor:1});
      const errors=[];
      const row={id,errors,ready:false};result.push(row);
      page.on('pageerror',e=>{errors.push(e.message);console.log('PAGE_ERROR',e.message);});
      page.on('console',m=>{if(m.type()==='error')console.log('CONSOLE',m.text());});
      page.on('requestfailed',r=>{errors.push(r.url()+':'+r.failure().errorText);console.log('REQUEST_FAILED',r.url());});
      await page.goto(pathToFileURL(path.join(input,'preview-'+id+'.html')).href);
      const frame=page.frameLocator('iframe');
      try{await frame.locator('#assembly-'+id+'[data-ready="true"]').waitFor({timeout:20000});}
      catch(e){console.log('ROOT',await frame.locator('#assembly-'+id).getAttribute('data-ready'));console.log('ERROR',await frame.locator('[data-error]').innerText());await page.screenshot({path:path.join(output,'failure.png')});throw e;}
      const canvas=frame.locator('canvas');
      await canvas.screenshot({path:path.join(output,'assembly-'+id+'.png')});
      await frame.locator('[data-angle]').click();
      await frame.locator('[data-legend] input').first().uncheck();
      await frame.locator('[data-legend] input').first().check();
      if(id==='07009'){
        await frame.locator('[data-next]').click();
        await frame.locator('[data-next]').click();
        if(await frame.locator('#assembly-'+id).getAttribute('data-frame')!=='80')throw Error('next_step_failed');
        await frame.locator('[data-prev]').click();
        if(await frame.locator('#assembly-'+id).getAttribute('data-frame')!=='0')throw Error('previous_step_failed');
        await frame.locator('[data-play]').click();
        await page.waitForTimeout(1300);
        const playingFrame=Number(await frame.locator('#assembly-'+id).getAttribute('data-frame'));
        if(playingFrame<=0)throw Error('play_did_not_advance');
        await frame.locator('[data-play]').click();
        await canvas.screenshot({path:path.join(output,'07009-moving.png')});
        // Keyboard input exercises the real range handler for each separation step.
        const slider=frame.locator('[data-progress]');
        await slider.focus();await slider.press('Home');
        for(let i=0;i<4;i++){
          await slider.fill(String(i*80+79));await slider.dispatchEvent('input');
          await canvas.screenshot({path:path.join(output,'07009-step-'+(i+1)+'.png')});
        }
        await slider.press('End');
        const text=await frame.locator('[data-status]').innerText();
        if(!text.includes('완료')||!text.includes('3'))throw Error('terminal_state_missing');
        await canvas.screenshot({path:path.join(output,'07009-terminal.png')});
      }
      await page.setViewportSize({width:360,height:1100});
      await frame.locator('[data-original]').click();
      await page.screenshot({path:path.join(output,'mobile-'+id+'.png')});
      const overflow=await frame.locator('#assembly-'+id).evaluate(r=>r.scrollWidth>r.clientWidth+1);
      if(overflow)throw Error('mobile_overflow');
      Object.assign(row,{mobile_overflow:overflow,ready:true,playback_checked:id==='07009'});
      await page.close();
    }
  } finally {await browser.close();fs.writeFileSync(path.join(output,'browser_check.json'),JSON.stringify(result,null,2),{flag:'wx'});}
  if(result.some(r=>r.errors.length))throw Error('browser_errors');
  console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;});
