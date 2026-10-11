
import { useEffect, useState } from "react";

type Champion = {
  id: string;
  name: string;
  title: string;
  tags: string[];
};

function App() {
  const [champions, setChampions] = useState<Champion[]>([]);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    async function fetchChampions() {
      try {
        const response = await fetch(
          "http://127.0.0.1:8000/champions"
        );

        if (!response.ok) {
          throw new Error("Failed to fetch champions");
        }

        const data = await response.json();
        setChampions(data.champions);
      } catch {
        setError("Could not load champion data.");
      }
    }

    fetchChampions();
  }, []);

  const filteredChampions = champions.filter((champion) =>
    champion.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <main>
      <h1>RiftPredict</h1>
      <p>AI-Powered League of Legends Analytics</p>

      <input
        type="text"
        placeholder="Search for a champion..."
        value={search}
        onChange={(event) => setSearch(event.target.value)}
      />

      {error && <p>{error}</p>}

      <h2>Champions ({filteredChampions.length})</h2>

      {filteredChampions.map((champion) => (
        <div key={champion.id}>
          <h3>{champion.name}</h3>
          <p>{champion.title}</p>
          <p>Classes: {champion.tags.join(", ")}</p>
        </div>
      ))}
    </main>
  );
}

export default App;
