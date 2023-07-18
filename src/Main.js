import './Main.scss';
import { Helmet, HelmetProvider } from 'react-helmet-async';
import LOGO from './AICAMP.png';

function App() {
  return (
    <div className="App">
      <HelmetProvider>
        <Helmet>
          <title>Basic CV App</title>
          <link rel="icon" href={LOGO} alt="4079" />
        </Helmet>
      </HelmetProvider>
      
      <img id="Image1" href={LOGO}/>
      <div id="body">
        
      <p id="Title">Computer Vision Project</p>
        <div id="Text1">
          <form onClick={e => e.preventDefault()}>
            <input type="file" id="file" accept="image/*" />

            <div id="result"></div>

            <button type="submit" id="btn">
              Submit
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
