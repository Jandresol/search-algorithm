import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [profiles, setProfiles] = useState([]);
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [search, setSearch] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/profiles.json")
      .then((res) => res.json())
      .then((data) => {
        setProfiles(data);
        setSelectedProfile(data[0]); // default to first profile
      });
  }, []);

  const handleProfileChange = (id) => {
    const profile = profiles.find((p) => p.id === parseInt(id));
    setSelectedProfile(profile);
  };

  const handleSearch = async () => {
    if (!search.trim() || !selectedProfile) return;
    setLoading(true);
    const res = await fetch("http://localhost:8000/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ search_prompt: search, user_profile: selectedProfile }),
    });
    const data = await res.json();
    setResults(data.recommendations);
    setLoading(false);
  };

  const handleSuggestions = async () => {
    if (!selectedProfile) return;
    setLoading(true);
    const res = await fetch("http://localhost:8000/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ search_prompt: "", user_profile: selectedProfile }),
    });
    const data = await res.json();
    setResults(data.recommendations);
    setLoading(false);
  };

  return (
    <div className="app">
      <h1>Volunteer Opportunity Finder</h1>

      {/* <label htmlFor="profile-select">Select Profile:</label> */}
        <div className="profile-cards">
          {profiles.map((profile) => (
            <div
              key={profile.id}
              className={`profile-card ${selectedProfile?.id === profile.id ? "selected" : ""}`}
              onClick={() => handleProfileChange(profile.id)}
            >
              <h3>{profile.name}</h3>
                <p><strong>Skills:</strong> {profile.skills.join(", ")}</p>
                <p><strong>Training:</strong> {profile.training.join(", ")}</p>
                <p><strong>Interests:</strong> {profile.interests.join(", ")}</p>
            </div>
          ))}
        </div>

      <div className="search-box">
        <input
          type="text"
          placeholder="Describe what you're looking for..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <button onClick={handleSearch}>Search</button>
        <button onClick={handleSuggestions}>User Suggestions</button>
      </div>

      {loading && <p className="loading">Searching...</p>}

      <div className="results">
        {results.map((r) => (
          <div className="card" key={r.id}>
            <h2>{r.name}</h2>
            <p>{r.description.slice(0, 100)}...</p>
            <p className="score">Score: {r.score.toFixed(4)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
