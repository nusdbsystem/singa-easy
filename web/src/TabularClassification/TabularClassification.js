import React from "react";
import axios from 'axios';

import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import Divider from '@material-ui/core/Divider';
import TextField from "@material-ui/core/TextField"
import Typography from "@material-ui/core/Typography"
import { withStyles } from "@material-ui/core/styles"

import PropTypes from "prop-types"
import { compose } from "redux"
import ReactEcharts from 'echarts-for-react';

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
    },
    myForm: {
        marginRight: 20,
        width: '25ch',
    },
    textField: {
        marginRight: 20,
        alignItems: 'center',
    }
})
class TabularClassification extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired
    }

    state = {
        url: "",
        results: "",
        answerReturned: false,
        FormIsValid: false,
        inputList: [{ variable: "", value: "" }],
        predictionResp: [],
        emptyFields: false,
        errorMsg:"Please fill in this field",
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
        {!this.state.FormIsValid && this.setState({emptyFields: true}) && alert('Please fill in all fields.')}
        console.log(this.state.errorMsg)
        const formData = this.state.inputList

        console.log(JSON.stringify(formData))

        // try {
        //     const res = await axios.post(
        //         `${this.state.url}`,
        //         formData
        //     );
        //     console.log("file uploaded, axios res.data: ", res.data)
        //     console.log("axios full response schema: ", res)
        //     this.setState(prevState => ({
        //         results: res.data,
        //         answerReturned: true
        //     }))
        // } catch (err) {
        //     console.error(err, "error")
        //     this.setState({
        //         message: "Upload failed"
        //     })
        // }
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
    checkIsInvalid = (value) => {
        if (!value)
        {return this.state.errorMsg}
        else
        {
            return
        }
    }
    handleInputChange = (e, index) => {
        const { name, value } = e.target;
        const list = this.state.inputList;
        list[index][name] = value
        this.setState({ inputList: list })
    }
    handleAddClick = (e, index) => {
        this.setState({emptyFields: false});
        const list = this.state.inputList;
        list.push({ variable: "", value: "" })
        console.log(list);
        this.setState({ inputList: list })
    }
    handleRemoveClick = index => {
        this.setState({emptyFields: false});
        const list = this.state.inputList;
        list.splice(index, 1)
        this.setState({ inputList: list })
    }
    getOption = (predictionResp) => {
        var seriesdata = predictionResp.map(item => (item.mean * 100).toFixed(2))
        return {
            title: { text: 'Prediction Results' },
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            xAxis: { type: 'value', boundaryGap: [0, 0.01], axisLabel: { fontSize: 14 } },
            yAxis: { type: 'category', data: predictionResp.map(item => item.label), axisLabel: { fontSize: 14 } },
            series: { type: 'bar', data: seriesdata, label: { show: true, position: 'inside', formatter: "{c}%" } },
            textStyle: { fontWeight: 800, fontSize: 16 },
        }
    };

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
                        Fill in the form to query
                  </Typography>
                    <form method="POST" id="myForm" className="myForm" align="center" noValidate autoComplete="off">
                        {this.state.inputList.map((x, i) => {
                            return (
                                <div className="box" align="center">
                                    <TextField error={this.checkIsInvalid(x.variable) && this.state.emptyFields} helperText={this.state.emptyFields && x.variable === "" ? 'Please fill in this field' : ''} className="textField" name="variable" variant="outlined" value={x.variable} onChange={e => this.handleInputChange(e, i)} required/>
                                    <TextField error={this.checkIsInvalid(x.value) && this.state.emptyFields} helperText={this.state.emptyFields && x.value === "" ? 'Please fill in this field' : ''} className="textField" name="value" variant="outlined" value={x.value} onChange={e => this.handleInputChange(e, i)} required/>

                                    {this.state.inputList.length != 1 &&<Button variant="contained" color="secondary" onClick={() => this.handleRemoveClick(i)}>Delete</Button>}
                                    {this.state.inputList.length - 1 === i && <Button variant="contained" color="default" onClick={e => this.handleAddClick(e, i)}>Add more</Button>}

                                </div>

                            )
                        })}

                        <br />
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={this.handleCommit}
                            // disabled={
                            //     !this.state.FormIsValid}
                        >
                            Predict
                  </Button>
                    </form>
                </div>
                <div className={classes.contentWrapper}>

                    {this.state.answerReturned &&
                        <div className={classes.response}>
                            <Grid container spacing={3}>
                                <Grid item xs={12} sm={6}>
                                    <Typography variant="h5" gutterBottom align="center">
                                        Labels and percentage
                                    </Typography>
                                    {/* 
                                    **** TO BE UPDATED BASED ON MODEL RESPONSE ****
                                    <ReactEcharts
                                        option={this.getOption(this.state.predictionResp)}
                                    /> */}
                                </Grid>
                            </Grid>

                        </div>
                    }

                </div>
            </React.Fragment >
        )
    }
}
export default compose(withStyles(styles))(TabularClassification);
