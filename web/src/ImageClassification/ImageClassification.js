import React from "react"
import PropTypes from "prop-types"

import axios from 'axios';

// import { connect } from "react-redux"
import { compose } from "redux"
// import { push } from "connected-react-router"

import { withStyles } from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"
import Divider from '@material-ui/core/Divider';
import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';

import Grid from '@material-ui/core/Grid';

import FileDropzone from "../components/FileDropzone"
import UploadProgressBar from '../components/UploadProgressBar';

// // read query-string
// import queryString from 'query-string'

import ReactEcharts from 'echarts-for-react';
import { calculateGaussian } from "../components/calculateGaussian"

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

class ImageClassification extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired,
        handleHeaderTitleChange: PropTypes.func,
        resetLoadingBar: PropTypes.func,
        formState: PropTypes.string,
    }

    state = {
        predictorHost: "",
        FormIsValid: false,
        noPredictorSelected: false,
        // populate the files state from FileDropzone
        selectedFiles: [],
        uploadedImg: null,
        // post-upload status:
        message: "",
        uploadPercentage: 0,
        formState: "init",
        // populate the response
        predictionDone: false,
        gradcamImg: "",
        limeImg: "",
        mcDropout: [],
    }

    componentDidUpdate(prevProps, prevState) {
        // if form's states have changed
        if (
            this.state.selectedFiles !== prevState.selectedFiles
        ) {
            if (
                this.state.selectedFiles.length !== 0
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

    handleChange = (e) => {
        this.setState({ predictorHost: e.target.value });
    }

    onDrop = files => {
        console.log("onDrop called, acceptedFiles: ", files)
        const currentFile = files[0]
        const imgReader = new FileReader()
        imgReader.addEventListener("load", () => {
            this.setState({
                uploadedImg: imgReader.result
            })
        })
        imgReader.readAsDataURL(currentFile)
        this.setState({
            selectedFiles: files
        })
    }

    handleRemoveCSV = () => {
        this.setState({
            selectedFiles: []
        })
        console.log("file removed")
    }

    handleCommit = async e => {
        e.preventDefault();
        this.setState({
            // reset previous response, if any
            predictionDone: false,
            gradcamImg: "",
            limeImg: "",
            mcDropout: [],
            uploadPercentage: 0,
            FormIsValid: false,
            formState: "loading",
        })

        const formData = new FormData()
        console.log("selectedFiles[0]: ", this.state.selectedFiles[0])
        formData.append("img", this.state.selectedFiles[0])

        try {
            const res = await axios.post(
                `http://${this.state.predictorHost}`,
                // `http://panda.d2.comp.nus.edu.sg:54691/predict`,
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                    onUploadProgress: progressEvent => {
                        // progressEvent will contain loaded and total
                        let percentCompleted = parseInt(
                            Math.round((progressEvent.loaded * 100) / progressEvent.total)
                        )
                        console.log("From EventEmiiter, file Uploaded: ", percentCompleted)
                        this.setState({
                            uploadPercentage: percentCompleted
                        })
                    }
                }
            );
            // res.data is the object sent back from the server
            console.log("file uploaded, axios res.data: ", res.data)
            console.log("axios full response schema: ", res)

            this.setState(prevState => ({
                formState: "idle",
                message: "Upload and prediction done",
                predictionDone: true,
                gradcamImg: res.data.explanations.gradcam_img,
                limeImg: res.data.explanations.lime_img,
                mcDropout: res.data.mc_dropout
            }))
        } catch (err) {
            console.error(err, "error")
            this.setState({
                message: "Upload failed"
            })
        }
    }

    getOption = (mcDropout) => {
        console.log("mcDropout: ", mcDropout)

        return {
            title: {
                text: "MC Dropout",
                // x: "center"
            },
            // toolbox: {
            //   feature: {
            //     dataView: { show: true, readOnly: false },
            //     magicType: { show: true, type: ['line', 'bar'] },
            //     restore: { show: true },
            //     saveAsImage: { show: true }
            //   }
            // },
            legend: {
                data: mcDropout.map(item => item.label)
            },
            tooltip: {
                trigger: 'axis',
            },
            xAxis: {
                type: 'value',
                name: "Mean",
                nameLocation: 'middle',
                min: 0,
                max: 1
            },
            yAxis: {
                type: 'value',
                name: "Probability",
                min: 0,
                max: 1
            },
            series: mcDropout.map(item => {
                return {
                    name: item.label,
                    type: "line",
                    data: calculateGaussian(item.mean, item.std)
                }
            })
        }
    };

    getOption2 = (mcDropout) => {
        var seriesdata = mcDropout.map(item => (item.mean * 100).toFixed(2))
        return {
            title: { text: 'Prediction Results' },
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            xAxis: { type: 'value', boundaryGap: [0, 0.01] },
            yAxis: { type: 'category', data: mcDropout.map(item => item.label) },
            series: { type: 'bar', data: seriesdata, label: { show: true, position: 'inside', formatter: "{c}%" } }
        }
    };

    render() {

        const { classes } = this.props

        return (
            <React.Fragment>

                <div className={classes.contentWrapper}>
                    <Typography className={classes.pos} gutterBottom align="center">
                        Predictor Host: {this.state.predictorHost}
                    </Typography>
                    <form onSubmit={this.handleSubmit} align="center">
                        <div className="predhost">
                            <input type="text"
                                value={this.state.predictorHost}
                                onChange={this.handleChange}
                                className="form-control" />
                        </div>
                    </form>
                    <br />
                    <Divider />
                    <br />
                    <Typography variant="h5" gutterBottom align="center">
                        Upload Test Image
                  </Typography>
                    <FileDropzone
                        files={this.state.selectedFiles}
                        onCsvDrop={this.onDrop}
                        onRemoveCSV={this.handleRemoveCSV}
                        AcceptedMIMEtypes={`
                      image/jpeg,
                      image/jpg,
                      image/png
                    `}
                        MIMEhelperText={`
                    (Only image format will be accepted)
                    `}
                        UploadType={`Image`}
                    />
                    <br />
                    <Button
                        variant="contained"
                        color="primary"
                        onClick={this.handleCommit}
                        disabled={
                            !this.state.FormIsValid ||
                            this.state.formState === "loading"}
                    >
                        Predict
                  </Button>
                </div>
                <div className={classes.contentWrapper}>
                    <div className={classes.progbarStatus}>
                        
                        {this.state.formState === "loading" &&
                            <React.Fragment>
                                <LinearProgress color="secondary" />
                                <br />
                            </React.Fragment>
                        }
                        <UploadProgressBar
                            percentCompleted={this.state.uploadPercentage}
                            fileName={
                                this.state.selectedFiles.length !== 0
                                    ? this.state.selectedFiles[0]["name"]
                                    : ""
                            }
                            formState={this.state.formState}
                            dataset={this.state.newDataset}
                        />
                        <br />
                        <Typography component="p">
                            <b>{this.state.message}</b>
                            <br />
                        </Typography>
                    </div>
                    <br />
                    {this.state.predictionDone &&
                        <div className={classes.response}>
                            <Grid container spacing={3}>
                                <Grid item xs={12} sm={6}>
                                    <Typography variant="h5" gutterBottom align="center">
                                        Original Image:
                          </Typography>
                                    <img
                                        className={classes.explainImg}
                                        src={this.state.uploadedImg}
                                        alt="OriginalImg"
                                    />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <Typography variant="h5" gutterBottom align="center">
                                        Labels and percentage
                          </Typography>
                                    <ReactEcharts
                                        option={this.getOption2(this.state.mcDropout)}
                                    />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <Typography variant="h5" gutterBottom align="center">
                                        Gradcam Image:
                          </Typography>
                                    <img
                                        className={classes.explainImg}
                                        src={`data:image/jpeg;base64,${this.state.gradcamImg}`}
                                        alt="GradcamImg"
                                    />
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <Typography variant="h5" gutterBottom align="center">
                                        Lime Image:
                          </Typography>
                                    <img
                                        className={classes.explainImg}
                                        src={`data:image/jpeg;base64,${this.state.limeImg}`}
                                        alt="LimeImg"
                                    />
                                </Grid>
                            </Grid>
                            <br />
                            <Divider />
                            <br />
                            <ReactEcharts
                                option={this.getOption(this.state.mcDropout)}
                                style={{ height: 500 }}
                            />
                        </div>
                    }
                </div>
            </React.Fragment>
        )
    }
}

export default compose(withStyles(styles))(ImageClassification);