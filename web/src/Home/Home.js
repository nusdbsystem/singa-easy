import React, {Component} from "react";
import Button from '@material-ui/core/Button';
import history from '../history';
import "./Home.css"

export default class Home extends Component {
    render() {
        return (
            <div className ="Home">
                <div className="lander">
                    <h1>Predictions</h1>
                    <p>Select from models below</p>
                    <form>
                    
                        <Button variant="contained" color="primary" onClick={() => history.push('/ImageClassification')}>Image Classification</Button><br/><br/>
                        <Button variant="contained" color="primary" onClick={() => history.push('/PosTagging')}>Pos Tagging</Button><br/><br/>
                        <Button variant="contained" color="primary" onClick={() => history.push('/TabularClassification')}>Tabular Classification</Button><br/><br/>
                        <Button variant="contained" color="primary" onClick={() => history.push('/TabularRegression')}>Tabular Regression</Button><br/><br/>
                        <Button variant="contained" color="primary" onClick={() => history.push('/SpeechRecognition')}>Speech Recognition</Button><br/><br/>
                        <Button variant="contained" color="primary" onClick={() => history.push('/ObjectDetection')}>Object Detection</Button>
                    </form>
                </div>
            </div>
        );
    }
}