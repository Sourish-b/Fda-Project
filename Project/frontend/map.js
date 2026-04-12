function initMap() {
  const iframe = document.getElementById("map-frame") || document.getElementById("indiaMapFrame");
  if (!iframe) {
    return;
  }

  iframe.src = "../model/saved/india_map.html";
  iframe.addEventListener("load", () => {
    console.log("Map loaded successfully");
  });
}


function highlightState(stateName) {
  // Future enhancement: wire this to Folium click events to drive dashboard interactions.
  console.log("State selected from map: " + stateName);
  updateCharts(stateName);
  updateStatePanel(stateName);
}


document.addEventListener("DOMContentLoaded", () => {
  initMap();
});
