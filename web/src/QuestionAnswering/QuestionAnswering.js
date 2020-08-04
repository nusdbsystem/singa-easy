import React from "react";
import axios from 'axios';

import Button from '@material-ui/core/Button';
import Divider from '@material-ui/core/Divider';
import PropTypes from "prop-types"

import Typography from "@material-ui/core/Typography"
import { withStyles } from "@material-ui/core/styles"
import { compose } from "redux"

const styles = theme => ({
    block: {
        display: "block",
    },
    addDS: {
        marginRight: theme.spacing(1),
    },
    contentWrapper: {
        margin: "16px 16px",
        //position: "relative",
        minHeight: 200,
    },
    // for query-params
    pos: {
        marginBottom: 12,
        alignItems: 'center'
    },
    // for response display
    response: {
        flexGrow: 1,
        marginTop: "20px",
    },
    explainImg: {
        margin: "0 auto",
        width: "90%",
    },
    progbarStatus: {
        padding: 20,
        overflowWrap: "break-word"
    }
})
class QuestionAnswering extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired
    }

    state = {
        url: "",
        questionarea: "Covid19 Question",
        question: "",
        answer: "",
        results: "",
        answerReturned: false,
        FormIsValid: false
    }
    componentDidUpdate(prevProps, prevState) {
        // if form's states have changed
        if (
            this.state.question !== prevState.question
        ) {
            if (
                this.state.question.length !== 0
            ) {
                this.setState({
                    FormIsValid: true
                })
                // otherwise disable COMMIT button
            } else {
                this.setState({
                    FormIsValid: false
                })
            }
        }
    }
    handleChange = ({ target: { name, value } }) => {
        this.setState(prevState => ({
            ...this.setState,
            [name]: value
        }));
        console.log(this.state);
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
            allowWithoutGesture: true
        }).then(result => {
            console.log(result);
            if (result.state === 'prompt' || result.state === 'granted') {
                navigator.clipboard.readText().then(
                    clipText => {
                        // document.getElementById("url").value = clipText;
                        this.setState({ url: clipText });
                        console.log(this.state.url)
                    });
            }
            else { alert("Permission to access clipboard denied!") }
        })

    }

    render() {
        const { classes } = this.props
        return (
            <React.Fragment>
                <div className={classes.contentWrapper}>
                    <Typography className={classes.pos} gutterBottom align="center">
                        Predictor Host: {this.state.url}
                    </Typography>
                    <form onSubmit={this.handleSubmit} align="center">
                        <div className="predhost">
                            <input id="url"
                                name="url"
                                type="text"
                                value={this.state.url}
                                onChange={this.handleChange}
                                className="form-control" />
                        </div><br />
                        <Button variant="contained"
                            color="primary"
                            onClick={this.handleClick}>Paste link here</Button>
                    </form>
                    <br />
                    <Divider />
                    <br />
                    <Typography variant="h5" gutterBottom align="center">
                        Fill in the Question
                  </Typography>

                    <form method="POST" id="myForm" name="myForm" align="center">
                        <p>
                            <input
                                type="text"
                                name="question"
                                id="question"
                                className="form-control"
                                value={this.state.question}
                                onChange={this.handleChange}
                                required /></p>


                        <Button
                            variant="contained"
                            color="primary"
                            onClick={this.handleCommit}
                            disabled={
                                !this.state.FormIsValid}
                        >
                            Predict
                  </Button>
                    </form>
                </div>
                <div className={classes.contentWrapper}>

                    {this.state.answerReturned &&
                        <div className={classes.response}>
                            <iframe title="answer"
                                srcDoc={this.state.results}
                                width="100%"
                                height="500px"></iframe>


                        </div>
                    }
                </div>
            </React.Fragment >
        )
    }
}
export default compose(withStyles(styles))(QuestionAnswering);
