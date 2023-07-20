import './Main.scss';
import { Helmet, HelmetProvider } from 'react-helmet-async';
import LOGO from './AICAMP.png';

function App() {
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
      const formattedResult = result.map((lang) => `${lang.language}: ${lang.percentage}%`);
      document.getElementById('result').innerHTML = formattedResult.join('<br>');
    
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

      <img id="Image1" href={LOGO} />
      <div id="body">

        <p id="Title">Computer Vision Project</p>
        <div id="Text1">
          <form>
            <input type="file" id="file" accept="image/*" />

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
