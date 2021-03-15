import React from "react";
import "./Home.css";
import Particles from "react-particles-js";
import Grid from "@material-ui/core/Grid";

const particlesOptions = {
  particles: {
    line_linked: {
      enable: true,
      color: "#000000",
      width: 0.5
    },
    color: { value: "#000000" },
    number: {
      value: 100
    },
    size: { value: 5 }
  }
};

export default function() {
  return (
    <Grid container spacing={1}>
      <Grid item xs={12} className="Home">
        <div className="lander">
          <Particles className="particles" params={particlesOptions} />
          <h1>Predictions</h1>
          <p className="textItem">
            To insert a brief description of Singa-Easy Models and objective of
            application
          </p>
        </div>
      </Grid>
    </Grid>
  );
}
