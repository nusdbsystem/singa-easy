import React from "react";
import axios from 'axios';

import Button from '@material-ui/core/Button';
import Divider from '@material-ui/core/Divider';
import PropTypes from "prop-types"

import Typography from "@material-ui/core/Typography"
import { withStyles } from "@material-ui/core/styles"
import { compose } from "redux"
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';

const StyledTableCell = withStyles((theme) => ({
    head: {
        backgroundColor: theme.palette.common.black,
        color: theme.palette.common.white,
    },
    body: {
        fontSize: 14,
    },

}))(TableCell);

const StyledTableRow = withStyles((theme) => ({
    root: {
        '&:nth-of-type(odd)': {
            backgroundColor: theme.palette.action.hover,
        },
    },
}))(TableRow);

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
    progbarStatus: {
        padding: 20,
        overflowWrap: "break-word"
    },

    table: {
        maxWidth: 250,
        maxHeight: 200,
        stickyHeader: true
    }
})
class PosTagging extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired
    }

    state = {
        url: "",
        inputText: "",
        formData: "",
        answer: "",
        results: "",
        answerReturned: false,
        FormIsValid: false
    }
    componentDidUpdate(prevProps, prevState) {
        // if form's states have changed
        if (
            this.state.inputText !== prevState.inputText
        ) {
            if (
                this.state.inputText.length !== 0
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

        this.state.formData = this.state.inputText.split(" ")

        try {
            const res = await axios.post(
                `${this.state.url}`,
                this.state.formData
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
                        Input Text for Part-of-Speech Tagging
                  </Typography>

                    <form method="POST" id="myForm" name="myForm" align="center">
                        <p>
                            <input
                                type="text"
                                name="inputText"
                                id="inputText"
                                className="form-control"
                                value={this.state.inputText}
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
                <div id="labelledResp" className={classes.contentWrapper}>
                    {this.state.answerReturned === true &&
                        <TableContainer component={Paper}>
                            <Table className={classes.table} aria-label="customized table">
                                <TableHead>
                                    <TableRow>
                                        <StyledTableCell >Token</StyledTableCell>
                                        <StyledTableCell align="right">Tag</StyledTableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {this.state.results[0].map((result, idx) => (
                                        <StyledTableRow key={idx}>
                                            <StyledTableCell component="th" scope="row">
                                                {this.state.formData[idx]}
                                            </StyledTableCell>
                                            <StyledTableCell align="right">{result}</StyledTableCell>
                                        </StyledTableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    }
                </div>

            </React.Fragment >
        )
    }
}
export default compose(withStyles(styles))(PosTagging);
