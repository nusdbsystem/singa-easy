import React from "react";
import axios from "axios";

import Button from "@material-ui/core/Button";
import Divider from "@material-ui/core/Divider";
import TextField from "@material-ui/core/TextField";
import Typography from "@material-ui/core/Typography";
import MenuItem from "@material-ui/core/MenuItem";
import FormControl from "@material-ui/core/FormControl";
import Select from "@material-ui/core/Select";
import { withStyles } from "@material-ui/core/styles";
import PropTypes from "prop-types";
import { compose } from "redux";
import FileDropzone from "../components/FileDropzone";
import { CSVLink } from "react-csv";

const styles = theme => ({
  button: {
    backgroundColor: "#f7871e",
    color: "#fff",
    padding: "10px 15px",
    "&:hover": {
      backgroundColor: "#fa9b42"
    }
  },
  block: {
    display: "block"
  },
  addDS: {
    marginRight: theme.spacing(1)
  },
  contentWrapper: {
    margin: "120px 100px",
    //position: "relative",
    minHeight: 200
  },
  // for query-params
  pos: {
    marginBottom: 12,
    alignItems: "center"
  },
  // for response display
  response: {
    flexGrow: 1,
    marginTop: "20px"
  },
  explainImg: {
    margin: "0 auto",
    width: "90%"
  },
  progbarStatus: {
    padding: 20,
    overflowWrap: "break-word"
  },
  myForm: {
    marginRight: 20,
    width: "25ch"
  },
  textField: {
    marginRight: 20,
    alignItems: "center"
  }
});
class TabularRegression extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired
  };

  state = {
    url: "",
    singleAnswerReturned: false,
    multipleAnswerReturned: false,
    FormIsValid: false,
    uploadedFile: null,
    selectedFiles: [],
    inputList: [{ variable: "", value: "" }],
    predictionResp: [],
    emptyFields: false,
    errorMsg: "Please fill in this field",
    inputCount: 0,
    queryState: "single",
    csvFile: ""
  };

  onDrop = files => {
    console.log("onDrop called, acceptedFiles: ", files);
    const currentFile = files[0];
    const fileReader = new FileReader();
    fileReader.addEventListener("load", () => {
      this.setState({
        uploadedFile: fileReader.result
      });
    });
    fileReader.readAsText(currentFile);
    this.setState({
      selectedFiles: files
    });
  };

  handleRemoveCSV = () => {
    this.setState({
      selectedFiles: []
    });
    console.log("file removed");
  };

  handleChange = ({ target: { name, value } }) => {
    this.setState(prevState => ({
      ...this.setState,
      [name]: value
    }));
    console.log(this.state);
  };

  queryState = async e => {
    this.setState({ queryState: e.target.value });
  };

  handleCommit = async e => {
    e.preventDefault();
    const formData = this.state.inputList;
    if (!this.state.FormIsValid) {
      this.setState({ emptyFields: true });
    } else {
      var dict = {};
      for (var i = 0; i < formData.length; i++) {
        var valueInput = Number(formData[i].value);
        if (Number.isNaN(valueInput)) {
          dict[formData[i].variable] = formData[i].value;
        } else {
          dict[formData[i].variable] = Number(formData[i].value);
        }
      }
      console.log(dict);

      try {
        const res = await axios.post(`${this.state.url}`, dict);
        console.log("file uploaded, axios res.data: ", res.data);
        console.log("axios full response schema: ", res);
        this.setState(prevState => ({
          predictionResp: res.data[0],
          singleAnswerReturned: true
        }));
      } catch (err) {
        console.error(err, "error");
        this.setState({
          message: "Upload failed"
        });
      }
    }
  };

  handleBulkCommit = async e => {
    e.preventDefault();
    this.setState(prevState => ({
      predictionResp: []
    }));
    const csvMod = this.state.uploadedFile.replace(/\n/g, ";");
    const arr = csvMod.split(";");

    var dictArr = [];
    for (var i = 0; i < arr.length; i++) {
      var dict = {};
      var res = arr[i].split(",");
      for (var j = 0; j < res.length; j += 2) {
        var valueInput = Number(res[j + 1]);
        if (Number.isNaN(valueInput)) {
          dict[res[j]] = res[j + 1];
        } else {
          dict[res[j]] = Number(res[j + 1]);
        }
      }
      dictArr.push(dict);
    }

    for (i = 0; i < dictArr.length; i++) {
      try {
        const res = await axios.post(`${this.state.url}`, dictArr[i]);
        console.log("file uploaded, axios res.data: ", res.data);
        console.log("axios full response schema: ", res);
        this.setState(prevState => ({
          predictionResp: [...prevState.predictionResp, res.data[0]],
          multipleAnswerReturned: true
        }));

        arr[i] = arr[i].concat("," + String(res.data));
      } catch (err) {
        console.error(err, "error");
        this.setState({
          message: "Upload failed"
        });
      }
    }
    var csv = "";
    arr.forEach(function(row) {
      var arrRow = row.split(",");
      csv += arrRow.join(",");
      csv += "\n";
    });
    console.log(csv);
    this.setState({ csvFile: csv });
  };

  handleClick = e => {
    e.preventDefault();
    navigator.permissions
      .query({
        name: "clipboard-read",
        allowWithoutGesture: true
      })
      .then(result => {
        console.log(result);
        if (result.state === "prompt" || result.state === "granted") {
          navigator.clipboard.readText().then(clipText => {
            // document.getElementById("url").value = clipText;
            this.setState({ url: clipText });
            console.log(this.state.url);
          });
        } else {
          alert("Permission to access clipboard denied!");
        }
      });
  };
  checkIsInvalid = value => {
    if (!value) {
      return this.state.errorMsg;
    } else {
      return;
    }
  };
  handleInputChange = (e, index) => {
    const { name, value } = e.target;
    const list = this.state.inputList;
    list[index][name] = value;
    this.setState({ inputList: list });
    var count = 0;
    for (var i = 0; i < this.state.inputList.length; i++) {
      if (this.state.inputList[i].variable && this.state.inputList[i].value) {
        count += 1;
      }
    }
    if (count === this.state.inputList.length) {
      this.setState({ FormIsValid: true });
    } else {
      this.setState({ FormIsValid: false });
    }
  };
  handleAddClick = (e, index) => {
    this.setState({ emptyFields: false, FormIsValid: false });
    const list = this.state.inputList;
    list.push({ variable: "", value: "" });
    this.setState({ inputList: list });
  };
  handleRemoveClick = index => {
    this.setState({ emptyFields: false });
    const list = this.state.inputList;
    list.splice(index, 1);
    this.setState({ inputList: list });
  };

  render() {
    const { classes } = this.props;
    return (
      <React.Fragment>
        <div className={classes.contentWrapper}>
          <Typography className={classes.pos} gutterBottom align="center">
            Predictor Host: {this.state.url}
          </Typography>
          <form onSubmit={this.handleSubmit} align="center">
            <div className="predhost">
              <input
                id="url"
                name="url"
                type="text"
                value={this.state.url}
                onChange={this.handleChange}
                className="form-control"
              />
            </div>
            <br />
            <Button
              variant="contained"
              className={classes.button}
              onClick={this.handleClick}
            >
              Paste link here
            </Button>
          </form>
          <br />
          <Divider />
          <br />
          <div align="center">
            <Typography variant="h5" gutterBottom align="center">
              Select from single query (manual input) or multiple query (upload
              CSV file)
            </Typography>
            <br />
            <FormControl className="formControl">
              <Select value={this.state.queryState} onChange={this.queryState}>
                <MenuItem value={"single"}> Single</MenuItem>
                <MenuItem value={"multiple"}> Multiple</MenuItem>
              </Select>
            </FormControl>
            <br />
          </div>

          {this.state.queryState === "single" && (
            <div>
              <br />
              <Typography variant="h5" gutterBottom align="center">
                <b>FORM</b>
              </Typography>
              <form
                method="POST"
                id="myForm"
                className="myForm"
                align="center"
                noValidate
                autoComplete="off"
              >
                {this.state.inputList.map((x, i) => {
                  return (
                    <div className="box" align="center">
                      <TextField
                        error={
                          this.checkIsInvalid(x.variable) &&
                          this.state.emptyFields
                        }
                        helperText={
                          this.state.emptyFields && x.variable === ""
                            ? "Please fill in this field"
                            : ""
                        }
                        className="textField"
                        name="variable"
                        variant="outlined"
                        value={x.variable}
                        onChange={e => this.handleInputChange(e, i)}
                        required
                      />
                      <TextField
                        error={
                          this.checkIsInvalid(x.value) && this.state.emptyFields
                        }
                        helperText={
                          this.state.emptyFields && x.value === ""
                            ? "Please fill in this field"
                            : ""
                        }
                        className="textField"
                        name="value"
                        variant="outlined"
                        value={x.value}
                        onChange={e => this.handleInputChange(e, i)}
                        required
                      />

                      {this.state.inputList.length !== 1 && (
                        <Button
                          variant="contained"
                          className={classes.button}
                          onClick={() => this.handleRemoveClick(i)}
                        >
                          Delete
                        </Button>
                      )}
                      {this.state.inputList.length - 1 === i && (
                        <Button
                          variant="contained"
                          className={classes.button}
                          onClick={e => this.handleAddClick(e, i)}
                        >
                          Add more
                        </Button>
                      )}
                    </div>
                  );
                })}

                <br />
                <Button
                  variant="contained"
                  className={classes.button}
                  onClick={this.handleCommit}
                  disabled={!this.state.FormIsValid}
                >
                  Predict
                </Button>
              </form>
            </div>
          )}
          {this.state.queryState === "multiple" && (
            <div>
              <br />
              <Typography variant="h5" gutterBottom align="center">
                <b>UPLOAD CSV FILE</b>
              </Typography>
              <form
                method="POST"
                id="myForm"
                className="myForm"
                align="center"
                noValidate
                autoComplete="off"
              >
                <FileDropzone
                  files={this.state.selectedFiles}
                  onCsvDrop={this.onDrop}
                  onRemoveCSV={this.handleRemoveCSV}
                  AcceptedMIMEtypes={`text/csv, application/vnd.ms-excel, application/csv, text/x-csv, application/x-csv, text/comma-separated-values, text/x-comma-separated-values`}
                  MIMEhelperText={`
                    (Only CSV format will be accepted)
                    `}
                  UploadType={`CSV`}
                />
                <br />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={this.handleBulkCommit}
                  disabled={this.state.selectedFiles.length === 0}
                >
                  Predict
                </Button>
              </form>
            </div>
          )}
        </div>
        <div className={classes.contentWrapper}>
          {this.state.queryState === "single" &&
            this.state.predictionResp &&
            this.state.singleAnswerReturned && (
              <div className={classes.response}>
                <Typography variant="h5" gutterBottom align="center">
                  The predicted output is {this.state.predictionResp}.
                </Typography>
              </div>
            )}
          {this.state.queryState === "single" &&
            this.state.predictionResp == null &&
            this.state.singleAnswerReturned && (
              <div className={classes.response}>
                <Typography variant="h5" gutterBottom align="center">
                  No predictions returned
                </Typography>
              </div>
            )}
          {this.state.queryState === "multiple" &&
            this.state.predictionResp &&
            this.state.multipleAnswerReturned && (
              <div className={classes.response}>
                <CSVLink data={this.state.csvFile}>
                  <Typography variant="h5" gutterBottom align="center">
                    Download prediction result
                  </Typography>
                </CSVLink>
              </div>
            )}
          {this.state.queryState === "multiple" &&
            this.state.predictionResp == null &&
            this.state.multipleAnswerReturned && (
              <div className={classes.response}>
                <Typography variant="h5" gutterBottom align="center">
                  No predictions returned
                </Typography>
              </div>
            )}
        </div>
      </React.Fragment>
    );
  }
}
export default compose(withStyles(styles))(TabularRegression);
