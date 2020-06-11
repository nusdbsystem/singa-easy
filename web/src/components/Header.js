import React from 'react';
import Link from '@material-ui/core/Link';
import Breadcrumbs from '@material-ui/core/Breadcrumbs';
import history from '../history';

function handleClick(e) {
  console.info('Breadcrumb clicked');
  history.push(Link.href)
}

const Header = () => {
  return (
   <Breadcrumbs aria-label="breadcrumb">
      <nav>
        <Link href="/" onClick={handleClick}color ="inherit">Home</Link>
        {" | "}
        <Link href="/ImageClassification" onClick={handleClick}color ="inherit">Image Classification</Link>
        {" | "}
        <Link href="/QuestionAnswering" onClick={handleClick}color ="inherit">Question Answering</Link>
        {" | "}
        <Link href="/PosTagging" onClick={handleClick}color ="inherit">POS Tagging</Link>
        {" | "}
        <Link href="/TabularClassification" onClick={handleClick}color ="inherit">Tabular Classification</Link>
        {" | "}
        <Link href="/TabularRegression" onClick={handleClick}color ="inherit">Tabular Regression</Link>
        {" | "}
        <Link href="/SpeechRecognition" onClick={handleClick}color ="inherit">Speech Recognition</Link>
        {" | "}
        <Link href="/ObjectDetection" onClick={handleClick}color ="inherit">Object Detection</Link>
      </nav>
      </Breadcrumbs>
  );
};

export default Header;