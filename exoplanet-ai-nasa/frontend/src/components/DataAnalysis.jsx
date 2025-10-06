import { useEffect, useState } from "react";
import Plot from "react-plotly.js";

function DataAnalysis() {
  const [plot, setPlot] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/analysis")
      .then(res => res.json())
      .then(data => setPlot(data));
  }, []);

  return (
    <div>
      <h1>Data Analysis</h1>
      {plot && plot.type === "plotly" ? (
        <Plot data={JSON.parse(plot.data).data} layout={JSON.parse(plot.data).layout} />
      ) : (
        <p>Loading analysis...</p>
      )}
    </div>
  );
}
export default DataAnalysis;
