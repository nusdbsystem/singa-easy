import React from "react";
import axios from 'axios';

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
        answerReturned: false,
        FormIsValid: false,
        inputList: [{ variable: "", value: "" }],
        predictionResp: [],
        emptyFields: false,
        errorMsg: "Please fill in this field",
        inputCount: 0,
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
    const formData = this.state.inputList
    if (!this.state.FormIsValid) { this.setState({ emptyFields: true }) }
    else {
    var dict = {}
    for (var i=0; i<formData.length; i++) {
        var valueInput = Number(formData[i].value)
        if (Number.isNaN(valueInput)) {
            dict[formData[i].variable] = formData[i].value
        }
        else {
            dict[formData[i].variable] = Number(formData[i].value)
        }
        
    }
    console.log(dict)
    

    try {
        const res = await axios.post(
            `${this.state.url}`,
            dict
        );
        console.log("file uploaded, axios res.data: ", res.data)
        console.log("axios full response schema: ", res)
        this.setState(prevState => ({
            predictionResp: res.data[0],
            answerReturned: true
        }))
    } catch (err) {
        console.error(err, "error")
        this.setState({
            message: "Upload failed"
        })
    }
}}
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
    if (!value) { return this.state.errorMsg }
    else {
        return
    }
}
handleInputChange = (e, index) => {
    const { name, value } = e.target;
    const list = this.state.inputList;
    list[index][name] = value
    this.setState({ inputList: list })
    var count = 0
    for (var i = 0; i < this.state.inputList.length; i++) {
        if (this.state.inputList[i].variable && this.state.inputList[i].value) { count += 1; }
    }
    if (count === this.state.inputList.length) { this.setState({ FormIsValid: true }) }
    else { this.setState({ FormIsValid: false }) }
}
handleAddClick = (e, index) => {
    this.setState({ emptyFields: false, FormIsValid: false });
    const list = this.state.inputList;
    list.push({ variable: "", value: "" })
    this.setState({ inputList: list })
}
handleRemoveClick = index => {
    this.setState({ emptyFields: false });
    const list = this.state.inputList;
    list.splice(index, 1)
    this.setState({ inputList: list })
}
getOption = (predictionResp) => {
    
    var seriesdata = predictionResp.map(item => (item * 100).toFixed(2))
    return {
        title: { text: 'Prediction Results' },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        xAxis: [{ type: 'value', boundaryGap: [0, 0.01], axisLabel: { fontSize: 14 } }],
        yAxis: [{ type: 'category', data: [0,1], axisLabel: { fontSize: 14 } }],
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
                                <TextField error={this.checkIsInvalid(x.variable) && this.state.emptyFields} helperText={this.state.emptyFields && x.variable === "" ? 'Please fill in this field' : ''} className="textField" name="variable" variant="outlined" value={x.variable} onChange={e => this.handleInputChange(e, i)} required />
                                <TextField error={this.checkIsInvalid(x.value) && this.state.emptyFields} helperText={this.state.emptyFields && x.value === "" ? 'Please fill in this field' : ''} className="textField" name="value" variant="outlined" value={x.value} onChange={e => this.handleInputChange(e, i)} required />

                                {this.state.inputList.length !== 1 && <Button variant="contained" color="secondary" onClick={() => this.handleRemoveClick(i)}>Delete</Button>}
                                {this.state.inputList.length - 1 === i && <Button variant="contained" color="default" onClick={e => this.handleAddClick(e, i)}>Add more</Button>}

                            </div>

                        )
                    })}

                    <br />
                    <Button
                        variant="contained"
                        color="primary"
                        onClick={this.handleCommit}
                    >
                        Predict
                  </Button>
                </form>
            </div>
            <div className={classes.contentWrapper}>

                {this.state.predictionResp && this.state.answerReturned &&
                    <div className={classes.response}>
                                <Typography variant="h5" gutterBottom align="center">
                                   Labels and percentage
                                    </Typography>
                                    <ReactEcharts
                                        option={this.getOption(this.state.predictionResp)}
                                    />

                    </div>
                }
                {this.state.predictionResp == null && this.state.answerReturned &&
                    <div className={classes.response}>
                                <Typography variant="h5" gutterBottom align="center">
                                   No predictions returned
                                    </Typography>
                                    

                    </div>
                }

            </div>
        </React.Fragment >
    )
}
}
export default compose(withStyles(styles))(TabularClassification);
