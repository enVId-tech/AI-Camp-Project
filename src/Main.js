import './Main.scss';
import { Helmet, HelmetProvider } from 'react-helmet-async';
import React, { useEffect, useState } from 'react';
import LOGO from './AICAMP.png';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    // create the preview
    if (selectedFile) {
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreview(objectUrl);

      // free memory when this component is unmounted
      return () => URL.revokeObjectURL(objectUrl);
    }
  }, [selectedFile]);

  const onSelectFile = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(null);
      setPreview(null);
      return;
    }

    setSelectedFile(e.target.files[0]);
  };

  const postFile = async () => {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];

    if (!file) {
      alert("Please select an image file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      const image_data = e.target.result.split(',')[1]; // Extract base64 data
      const payload = { image_data, filename: file.name };

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        alert(response.status);
        return;
      }

      const result = await response.json();
      console.log(result);
      if (result.length === 0) {
        alert('No language detected');
        return;
      }

      const resultDiv = document.getElementById('result');
      resultDiv.innerHTML = result.predicted_class;

    };

    reader.readAsDataURL(file);
  }

  return (
    <div className="App">
      <HelmetProvider>
        <Helmet>
          <title>Basic CV App</title>
          <link rel="icon" href={LOGO} alt="4079" />
        </Helmet>
      </HelmetProvider>

      <img id="Image1" href={LOGO} alt=""/>
      <div id="body">

        <p id="Title">Computer Vision Project</p>
        <div id="Text1">
          <form id="Form">
            <div id="inputs">
              <input type="file" id="file" accept="image/*" onChange={onSelectFile} />
              {selectedFile && <img id="preview" src={preview} alt="" />}
            </div>
            <div id="result"></div>

            <button type="submit" id="btn" onClick={e => { e.preventDefault(); postFile(); }}>
              Submit
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
