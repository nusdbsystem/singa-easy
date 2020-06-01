import React from "react";

class QuestionAnswering extends React.Component {
    state = {
        url: "",
        questionarea: "",
        question: "",
        answer: "",
    }

    handleChange = (e) => {
        this.setState({ url: e.target.value });
    }
    handleCommit = (e) => {
        e.preventDefault();
        var data = document.forms["myForm"]
        var s = {
            "Task": {
                "area": data[0].value,
                "questions": [data[1].value]
            }
        }

        console.log(JSON.stringify(s))

        var xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = function () {
            if (xmlhttp.readyState === XMLHttpRequest.DONE) {   // XMLHttpRequest.DONE == 4
                console.log(xmlhttp.response)
                if (xmlhttp.status === 200) {
                    document.getElementById("results_area").innerHTML = xmlhttp.responseText;
                }
                else if (xmlhttp.status === 400) {
                    alert('There was an error 400');
                }
                else {
                    alert('something else other than 200 was returned');
                }
            }
        };
        xmlhttp.open("POST", this.state.url, true);
        xmlhttp.send(JSON.stringify(s));

    }
    render() {
        return (
            <React.Fragment>
                <div className="QuestionAnswering">
                    <body>
                        <nav className="navbar navbar-expand navbar-light bg-light flex-column flex-md-row pipe-navbar justify-md-content-between" />
                        <a className="navbar-brand" href="https://www.comp.nus.edu.sg"><img src="https://logos-download.com/wp-content/uploads/2016/12/National_University_of_Singapore_logo_NUS_logotype.png" width="35" height="45" className="d-inline-block" /> COVID-19 Question Answering engine </a>
                    </body>
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
                            </p>

                            <div className="col-m-8">
                                <form method="POST" id="myForm" name="myForm">
                                    <p><label for="area">Question Area </label>
                                        <input type="text" name="area" id="area" className="myarea" value="Covid19 Question" /></p>

                                    <p><label for="question">Question </label>
                                        <input type="text" name="question" id="question" className="myquestion" required /></p>

                                    <input id="file-submit" value="Submit" className="btn btn-primary" type="button" onClick={this.handleCommit} />
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
                <div id='results_area' className="container-fluid ibm-code">
                    <div className="card-body">
                    </div>
                </div>
            </React.Fragment >
        )
    }
}
export default QuestionAnswering;
