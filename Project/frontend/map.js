function initMap() {
  const iframe = document.getElementById("indiaMapFrame");
  if (!iframe) return;

  // The "?t=" forces the browser to load the NEW map instead of the old cached gray one!
  iframe.src = "/model/saved/india_map.html?t=" + new Date().getTime();
}

// This receives the click event from our Folium map injection
window.highlightState = function(stateName) {
  console.log("State selected from map: " + stateName);
  
  // 1. Update the dropdown menu visually
  const selectEl = document.getElementById('stateSelect');
  if(selectEl) {
      selectEl.value = stateName;
  }

  // 2. Tell your charts.js file to update the dashboard
  if (typeof updateCharts === "function") updateCharts(stateName);
  if (typeof updateStatePanel === "function") updateStatePanel(stateName);
};

document.addEventListener("DOMContentLoaded", initMap);