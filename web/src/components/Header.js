import React from "react";
import { Tabs, Tab, AppBar } from "@material-ui/core";
import { Route, BrowserRouter, NavLink, Link } from "react-router-dom";
import Routes from "../Routes";
import { makeStyles } from "@material-ui/core/styles";
import Grid from "@material-ui/core/Grid";

const useStyles = makeStyles({
  root: {
    flexGrow: 1,
    backgroundColor: `#01213f`,
    textDecoration: `none`
  },
  paper: {
    fontSize: `25px`,
    backgroundColor: `#01213f`,
    padding: `20px 10px`,
    textAlign: "center",
    color: `#c5c3c3`,
    textTransform: `capitalize`,
    "&:hover": {
      backgroundColor: "#1890ff !important",
      color: `white`
    }
  },

  navHome: {
    padding: `25px 60px 25px 80px`,
    margin: `0px`,
    fontSize: `30px`,
    "&:hover": {
      textDecoration: `none`,
      backgroundColor: "#1890ff !important"
    }
  },

  navHomeText: {
    color: `#ffffff`,
    "&:hover": {
      textDecoration: `none`,
      backgroundColor: "#1890ff !important",
      color: `#ffffff`
    }
  },

  navItem: {
    padding: `15px`,
    color: `#c5c3c3`,
    fontSize: `15px`,
    "&:hover": {
      textDecoration: `none`
    }
  },
  navText: {
    fontSize: `15px`,
    padding: `17px 30px`,
    "&:hover": {
      textDecoration: `none`,
      color: `#ffffff`,
      backgroundColor: "#1890ff !important"
    }
  }
});

export default function Header() {
  const classes = useStyles();
  const routes = [
    "/ImageClassification",
    "/ObjectDetection",
    "/QuestionAnswering",
    "/PosTagging",
    "/TabularClassification",
    "/TabularRegression"
  ];
  return (
    <div className="App">
      <BrowserRouter>
        <Route
          path="/"
          render={history => (
            <AppBar className={classes.root}>
              <Grid container spacing={1}>
                <Grid item xs={2}>
                  <nav className={classes.navHome}>
                    <NavLink to="/home" className={classes.navHomeText}>
                      Singa-Easy
                    </NavLink>
                  </nav>
                </Grid>
                <Grid item xs={9}>
                  <Tabs
                    className={classes.navItem}
                    value={
                      history.location.pathname !== "/"
                        ? history.location.pathname
                        : false
                    }
                  >
                    {console.log(history.location.pathname)}
                    <Tab
                      className={classes.navText}
                      value={routes[0]}
                      label="Image Classification"
                      component={Link}
                      to={routes[0]}
                    />
                    <Tab
                      className={classes.navText}
                      value={routes[1]}
                      label="Object Detection"
                      component={Link}
                      to={routes[1]}
                    />
                    <Tab
                      className={classes.navText}
                      value={routes[2]}
                      label="Question Answering"
                      component={Link}
                      to={routes[2]}
                    />
                    <Tab
                      className={classes.navText}
                      value={routes[3]}
                      label="Pos Tagging"
                      component={Link}
                      to={routes[3]}
                    />
                    <Tab
                      className={classes.navText}
                      value={routes[4]}
                      label="Tabular Classification"
                      component={Link}
                      to={routes[4]}
                    />
                    <Tab
                      className={classes.navText}
                      value={routes[5]}
                      label="Tabular Regression"
                      component={Link}
                      to={routes[5]}
                    />
                  </Tabs>
                </Grid>
              </Grid>
            </AppBar>
          )}
        />
        <Routes />
      </BrowserRouter>
    </div>
  );
}
