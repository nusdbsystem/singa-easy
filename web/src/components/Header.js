import React, { useState, useEffect } from "react";
import {
  Tabs,
  Tab,
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemText
} from "@material-ui/core";
import { Route, BrowserRouter, NavLink, Link } from "react-router-dom";
import Routes from "../Routes";
import { makeStyles } from "@material-ui/core/styles";
import Grid from "@material-ui/core/Grid";
import MenuIcon from "@material-ui/icons/Menu";

const useStyles = makeStyles({
  root: {
    flexGrow: 1,
    backgroundColor: "#01213f",
    textDecoration: "none"
  },
  navHome: {
    padding: "30px 14px",
    textAlign: "center",
    margin: "0px 15px",
    "&:hover": {
      textDecoration: "none",
      backgroundColor: "#1890ff !important"
    }
  },
  navHomeText: {
    color: "#ffffff",
    fontSize: "22px",
    "&:hover": {
      textDecoration: "none",
      backgroundColor: "#1890ff !important",
      color: "#ffffff"
    }
  },
  navItem: {
    padding: "15px",
    color: "#c5c3c3",
    fontSize: "15px",
    "&:hover": {
      textDecoration: "none"
    }
  },
  navText: {
    fontSize: "15px",
    padding: "17px 30px",
    "&:hover": {
      textDecoration: "none",
      color: "#ffffff",
      backgroundColor: "#1890ff !important"
    }
  },

  drawerContainer: {
    paddingTop: "30px",
    textDecoration: "none",
    listStyle: "none"
  },
  drawerItem: {
    // fontSize: "40px",
    color: "#01213f",
    padding: "20px 40px",
    "&:hover": {
      textDecoration: "none",
      color: "#1890ff",
      backgroundColor: "#D5D5D5"
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

  const [state, setState] = useState({
    mobileView: false,
    drawerOpen: false
  });
  const { mobileView, drawerOpen } = state;

  useEffect(() => {
    const setResponsiveness = () => {
      return window.innerWidth < 1100
        ? setState(prevState => ({ ...prevState, mobileView: true }))
        : setState(prevState => ({ ...prevState, mobileView: false }));
    };
    setResponsiveness();
    window.addEventListener("resize", () => setResponsiveness());
  }, []);

  const displayMobile = () => {
    const handleDrawerOpen = () =>
      setState(prevState => ({ ...prevState, drawerOpen: true }));
    const handleDrawerClose = () =>
      setState(prevState => ({ ...prevState, drawerOpen: false }));
    return (
      <BrowserRouter>
        <Toolbar className={classes.root}>
          <IconButton
            {...{
              edge: "end",
              color: "secondary",
              "aria-label": "menu",
              "aria-haspopup": "true",
              onClick: handleDrawerOpen
            }}
          >
            <MenuIcon style={{ fontSize: "40px" }} />
          </IconButton>
          <Drawer
            {...{
              anchor: "left",
              open: drawerOpen,
              // onClose: handleDrawerClose,
              onClick: handleDrawerClose
            }}
          >
            <Route
              path="/"
              render={history => (
                <List className={classes.drawerContainer}>
                  <ListItem
                    button
                    className={classes.drawerItem}
                    component={Link}
                    to={routes[0]}
                  >
                    <ListItemText>Image Classification</ListItemText>
                  </ListItem>
                  <ListItem
                    button
                    className={classes.drawerItem}
                    component={Link}
                    to={routes[1]}
                  >
                    <ListItemText>Object Detection</ListItemText>
                  </ListItem>
                  <ListItem
                    button
                    className={classes.drawerItem}
                    component={Link}
                    to={routes[2]}
                  >
                    <ListItemText>Question Answering</ListItemText>
                  </ListItem>
                  <ListItem
                    button
                    className={classes.drawerItem}
                    component={Link}
                    to={routes[3]}
                  >
                    <ListItemText>POS Tagging</ListItemText>
                  </ListItem>
                  <ListItem
                    button
                    className={classes.drawerItem}
                    component={Link}
                    to={routes[4]}
                  >
                    <ListItemText>Tabular Classification</ListItemText>
                  </ListItem>
                  <ListItem
                    button
                    className={classes.drawerItem}
                    component={Link}
                    to={routes[5]}
                  >
                    <ListItemText>Tabular Regression</ListItemText>
                  </ListItem>
                </List>
              )}
            />
          </Drawer>
          <nav className={classes.navHome}>
            <NavLink to="/" className={classes.navHomeText}>
              Singa-Easy
            </NavLink>
          </nav>
        </Toolbar>
        <Routes />
      </BrowserRouter>
    );
  };
  const displayDesktop = () => {
    return (
      <BrowserRouter>
        <Route
          path="/"
          render={history => (
            <AppBar className={classes.root}>
              <Grid container spacing={2}>
                <Grid item xs={2}>
                  <nav className={classes.navHome}>
                    <NavLink to="/" className={classes.navHomeText}>
                      Singa-Easy
                      {/* <Typography  variant="h6" component="h1">Singa-Easy</Typography> */}
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
    );
  };

  return <div>{mobileView ? displayMobile() : displayDesktop()}</div>;
}
