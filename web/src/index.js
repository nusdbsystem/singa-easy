import React from "react";
import ReactDOM from "react-dom";
import Header from "./components/Header";

const rootElement = document.getElementById("root");
ReactDOM.render(
  <React.StrictMode>
    <Header />
  </React.StrictMode>,
  rootElement
);
