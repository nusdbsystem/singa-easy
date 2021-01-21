import React from "react";
import { Switch, Route } from "react-router-dom";

import Home from "./Home/Home";
import ImageClassification from "./ImageClassification/ImageClassification";
import ObjectDetection from "./ObjectDetection/ObjectDetection";
import QuestionAnswering from "./QuestionAnswering/QuestionAnswering";
import PosTagging from "./PosTagging/PosTagging";
import TabularClassification from "./TabularClassification/TabularClassification";
import TabularRegression from "./TabularRegression/TabularRegression";

export default function Routes() {
  return (
    <Switch>
      <Route path="/home" component={Home} />
      <Route path="/ImageClassification" component={ImageClassification} />
      <Route path="/ObjectDetection" component={ObjectDetection} />
      <Route path="/QuestionAnswering" component={QuestionAnswering} />
      <Route path="/PosTagging" component={PosTagging} />
      <Route path="/TabularClassification" component={TabularClassification} />
      <Route path="/TabularRegression" component={TabularRegression} />
    </Switch>
  );
}
