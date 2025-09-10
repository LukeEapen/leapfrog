// Reusable Help widget for POC 3b pages
(function(){
  if (window.__HelpWidgetInjected__) return; // singleton
  window.__HelpWidgetInjected__ = true;

  function h(html){ const tpl=document.createElement('template'); tpl.innerHTML=html.trim(); return tpl.content.firstChild; }

  const fab = h(`<button id="helpFab" title="Help" aria-label="Help"><span class="bi bi-life-preserver" aria-hidden="true"></span><span>Help</span></button>`);
  const panel = h(`
    <section id="helpPanel" role="dialog" aria-modal="false" aria-labelledby="helpTitle">
      <div id="helpHeader">
        <div id="helpTitle">Page Help</div>
        <button id="helpClose" aria-label="Close Help">×</button>
      </div>
      <div id="helpBody">
        <div id="helpIntro"></div>
        <h4>Business context</h4>
        <div id="helpBusiness"></div>
        <h4>What this page does</h4>
        <div id="helpPurpose"></div>
        <h4>Your actions here</h4>
        <ul id="helpActions"></ul>
      </div>
      <div id="helpFooter">
        <div class="chips" id="helpChips"></div>
        <div class="small text-secondary">Tip: Drag panel scroll to read more</div>
      </div>
    </section>`);

  function show(){ panel.classList.add('visible'); }
  function hide(){ panel.classList.remove('visible'); }

  fab.addEventListener('click', function(){
    if (panel.classList.contains('visible')) hide(); else show();
  });
  panel.querySelector('#helpClose').addEventListener('click', hide);

  document.addEventListener('keydown', function(e){ if(e.key==='Escape') hide(); });

  // init once DOM is ready
  function ready(fn){ if(document.readyState==='loading'){ document.addEventListener('DOMContentLoaded', fn); } else { fn(); } }

  ready(function(){
    document.body.appendChild(fab);
    document.body.appendChild(panel);
    // Populate per-page content if provided on window.HelpContent
    const cfg = window.HelpContent || {};
    const intro = cfg.intro || '';
    const business = cfg.business || '';
    const purpose = cfg.purpose || '';
    const actions = Array.isArray(cfg.actions) ? cfg.actions : [];
    const chips = Array.isArray(cfg.chips) ? cfg.chips : [];

    const introEl = document.getElementById('helpIntro'); if (intro) introEl.innerHTML = intro;
    const bizEl = document.getElementById('helpBusiness'); if (business) bizEl.innerHTML = business;
    const purpEl = document.getElementById('helpPurpose'); if (purpose) purpEl.innerHTML = purpose;
    const actEl = document.getElementById('helpActions'); if (actions.length) actEl.innerHTML = actions.map(a=>`<li>${a}</li>`).join('');
    const chipEl = document.getElementById('helpChips'); if (chips.length) chipEl.innerHTML = chips.map(c=>`<span class="help-chip">${c}</span>`).join('');

    // Auto-open on first visit to a page once per session key
    try{
      const key = 'help_seen_'+(document.title||location.pathname);
      if (!sessionStorage.getItem(key)) { show(); sessionStorage.setItem(key, '1'); }
    }catch(_){/* ignore */}
  });
})();
