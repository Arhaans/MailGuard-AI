import React, { useEffect, useState } from 'react';
import axios from 'axios';
import logo from './nillionimage.svg'; // Ensure this path is correct relative to your file structure
import "../App.css";

const Main = () => {
    const [items, setItems] = useState([]); // Stores search history
    const [inputValue, setInputValue] = useState(''); // Stores user input
    const [loading, setLoading] = useState(false); // Loading state for API call

    useEffect(() => {
        // Load items from local storage when the component mounts
        const storedItems = localStorage.getItem('items');
        if (storedItems) {
            setItems(JSON.parse(storedItems));
        }
    }, []);

    useEffect(() => {
        // Store items in local storage whenever they change
        localStorage.setItem('items', JSON.stringify(items));
    }, [items]);

    const handleInputChange = (e) => {
        setInputValue(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault(); // Prevent the default form submission
        if (!inputValue.trim()) return; // Exit if input is empty

        setLoading(true);
        try {
            // Use axios to make the POST request to the backend API
            const response = await axios.post('http://127.0.0.1:8000/classify-email/', {
                content: inputValue,
            });

            // Extract the classification result
            const classification = response.data.classification;
            const item = `${inputValue} - ${classification}`;

            // Add the result to the search history

            setItems((prevItems) => [...prevItems, item]);
        } catch (error) {
            console.error('Error connecting to the backend:', error);
        } finally {
            setInputValue(''); // Clear the input after submission
            setLoading(false);
        }
    };

    const clearHistory = () => {
        setItems([]); // Clear the items state
    };

    const deleteItem = (index) => {
        setItems((prevItems) => prevItems.filter((_, i) => i !== index));
    };

    return (
        <div className="py-10 min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
            <main className="flex flex-col items-center py-20 px-6 max-w-lg w-full bg-white rounded-lg shadow-lg">
                <div className="flex items-center space-x-4 mb-4">
                    <img 
                        src={logo} 
                        alt="Nillion Logo" 
                        className="w-12 h-12 animate-spin-slow" // CSS animation for infinite rotation
                    />
                    <h1 className="text-5xl font-extrabold text-indigo-700 tracking-wide text-center">
                        MailGuard AI
                    </h1>
                </div>
                <p className="text-lg text-gray-500 mb-10 text-center main-text">Powered by Nillion</p>
                
                <form className="flex items-center w-full max-w-md mb-8 space-x-2 main-form" onSubmit={handleSubmit}>
                    <input 
                        type="text" 
                        className="flex-grow p-3 border border-gray-400 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 input-text" 
                        placeholder="Ask me anything..."
                        value={inputValue}
                        onChange={handleInputChange}
                    />
                    <button 
                        type="submit" 
                        className="bg-indigo-600 text-white px-5 py-3 rounded-md shadow-md hover:bg-indigo-700 transition-all duration-300 ease-in-out transform hover:scale-105"
                        disabled={loading} // Disable button while loading
                    >
                        {loading ? 'Classifying...' : 'Submit'}
                    </button>
                </form>
                
                <div className="flex flex-col items-center mt-6">
                    <p className="text-gray-500">Need help?</p>
                    <div className="flex space-x-6 mt-3">
                        <a href="#" className="text-indigo-600 hover:text-indigo-500 hover:underline transition-all duration-200">Help Center</a>
                        <a href="#" className="text-indigo-600 hover:text-indigo-500 hover:underline transition-all duration-200">Contact Support</a>
                    </div>
                </div>

                {/* Search History Section */}
                <div className="mt-8 w-full">
                    <h2 className="text-lg font-bold text-indigo-600 text-center">Search History</h2>
                    <ul className="mt-4 space-y-2 text-center border rounded-lg bg-gray-50 p-4 shadow-md">
                        {items.length > 0 ? (
                            items.map((item, index) => (
                                <li key={index} className="flex justify-between items-center text-gray-700 border-b pb-2 hover:bg-indigo-100 transition-colors duration-200 cursor-pointer">
                                    <span>{item}</span>
                                    <button onClick={() => deleteItem(index)} className="text-red-600 hover:text-red-800">
                                        &times; {/* Cross symbol for deletion */}
                                    </button>
                                </li>
                            ))
                        ) : (
                            <li className="text-gray-400">No search history available.</li>
                        )}
                    </ul>
                    {items.length > 0 && (
                        <button 
                            onClick={clearHistory} 
                            className="mt-4 bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-all duration-300"
                        >
                            Clear History
                        </button>
                    )}
                </div>
            </main>
        </div>
    );
};

export default Main;
