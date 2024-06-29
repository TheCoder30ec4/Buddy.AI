import React, { useState } from 'react';
import './InputComponent.css';

function InputComponent() {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");

    const raw = JSON.stringify({
      Data: inputValue  // Use inputValue here
    });

    const requestOptions = {
      method: "POST",
      headers: myHeaders,
      body: raw,
      redirect: "follow"
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/input", requestOptions);
      const result = await res.json(); // Assuming the response is JSON
      setMessages((prevMessages) => [...prevMessages, { from: 'user', text: inputValue }, { from: 'bot', text: result.message }]);
      setInputValue('');
    } catch (error) {
      console.error('Error:', error);
      setMessages((prevMessages) => [...prevMessages, { from: 'user', text: inputValue }, { from: 'bot', text: 'Error: Unable to submit data' }]);
      setInputValue('');
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.from}`}>
            {message.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Enter text"
        />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
}

export default InputComponent;
