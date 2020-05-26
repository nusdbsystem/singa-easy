import React, {Component} from "react";
import {Router, Switch, Route} from "react-router-dom";

import Home from './Home/Home';
import ImageClassification from "./ImageClassification/ImageClassification";
import history from './history';

export default class Routes extends Component {
    render() {
        return(
            <Router history={history}>
                <Switch>
                    <Route path="/" exact component={Home} />
                    <Route path="/ImageClassification" component={ImageClassification} />
                    {/* <Route path="/PosTagging" component={PosTagging} />
                    <Route path="/TabularClassification" component={TabularClassification} />
                    <Route path="/TabularRegression" component={TabularRegression} />
                    <Route path="/SpeechRecognition" component={SpeechRecognition} />
                    <Route path="/ObjectDetection'" component={'ObjectDetection'} /> */}
                </Switch>
            </Router>
        )
    }
}