import React from "react";
import axios from 'axios';

import Button from '@material-ui/core/Button';

class QuestionAnswering extends React.Component {
    state = {
        url: "",
        questionarea: "Covid19 Question",
        question: "",
        answer: "",
        results: "",
        answerReturned: false,
    }

    handleChange = ({ target: { name, value } }) => {
        this.setState(prevState => ({
            ...this.setState,
            [name]: value
        }));
    }

    handleCommit = async e => {
        e.preventDefault();

        const formData = {
            "Task": {
                "area": this.state.questionarea,
                "questions": [this.state.question]
            }
        }

        console.log(JSON.stringify(formData))

        try {
            const res = await axios.post(
                `${this.state.url}`,
                formData
            );
            console.log("file uploaded, axios res.data: ", res.data)
            console.log("axios full response schema: ", res)
            this.setState(prevState => ({
                results: res.data,
                answerReturned: true
            }))
        } catch (err) {
            console.error(err, "error")
            this.setState({
                message: "Upload failed"
            })
        }
    }


    handleClick = (e) => {
        e.preventDefault();
        navigator.permissions.query({
            name: 'clipboard-read',
            allowWithoutGesture: false
        }).then(result => {
            console.log(result);
            if (result.state === 'prompt' || result.state === 'granted' ) {
                navigator.clipboard.readText().then(
                    clipText => { 
                    document.getElementById("url").value = clipText;
                     });
            }
            else {alert("Permission to access clipboard denied!")}
        })

    }

    render() {
        return (
            <div className="QuestionAnsweringContainer">
                <div className="QuestionAnswering">
                    <nav className="navbar navbar-expand navbar-light bg-light flex-column flex-md-row pipe-navbar justify-md-content-between" />
                    <a className="navbar-brand" href="https://www.comp.nus.edu.sg"><img src="https://logos-download.com/wp-content/uploads/2016/12/National_University_of_Singapore_logo_NUS_logotype.png" width="35" height="45" className="d-inline-block" alt="" /> COVID-19 Question Answering engine </a>
                </div>
                <div className="container-fluid ibm-code">
                    <div className="row">
                        <div className="col-lg-12">
                            <div className="row">
                                <div className="col-sm-8">
                                    <div className="card">
                                        <h5 className="card-header h5">Welcome! Please ask question</h5>
                                    </div>
                                </div>
                            </div>

                            <p>
                                <input type="text" name="url" id="url" className="myurl" placeholder="input url from server" value={this.state.url} onChange=
                                    {this.handleChange} />
                                    <Button variant="contained"
                        color="primary"
                        onClick={this.handleClick}>Paste link here</Button>
                            </p>

                            <div className="col-m-8">
                                <form method="POST" id="myForm" name="myForm">
                                    <p><label htmlFor="area">Question Area </label>
                                        <input
                                            type="text"
                                            name="questionarea"
                                            id="area"
                                            className="myarea"
                                            value={this.state.questionarea}
                                            onChange={this.handleChange}
                                        /></p>

                                    <p><label htmlFor="question">Question </label>
                                        <input
                                            type="text"
                                            name="question"
                                            id="question"
                                            className="myquestion"
                                            value={this.state.question}
                                            onChange={this.handleChange}
                                            required /></p>

                                    <input id="file-submit" value="Submit" className="btn btn-primary" type="button" onClick={this.handleCommit} />
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="container-fluid ibm-code">

                    {this.state.answerReturned &&
                        <div className="card-body results">
                            <iframe title ="answer" 
                            srcDoc={this.state.results} 
                            width = "100%"
                            height = "500px"></iframe>


                        </div>
                    }
                </div>
            </div>
        )
    }
}
export default QuestionAnswering;
