import React, { Component } from "react";
import "./Home.css";
import Particles from 'react-particles-js';

const particlesOptions = {
    "particles": {
        "line_linked": {
            "enable": true,
            "color": "#615555",
            "width": 0.5

        },
        "color": { value: "#000000" },
        "number": {
            "value": 50
        },
        "size": { "value": 3 },

    }
};

export default class Home extends Component {
    render() {
        return (

            <div className="Home">
                <div className="lander">
                    <Particles className="particles" params={particlesOptions} />
                    <h1>Predictions</h1>
                    <p>To insert a brief description of Singa-Easy Models and objective of application</p>
                </div>
            </div>

        );
    }
}