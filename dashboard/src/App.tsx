import { Sidebar } from "./components/Sidebar";

function App() {
  return (
    <div className="flex min-h-screen bg-slate-950 text-white">
      <Sidebar />

      <main className="p-8">
        <h1 className="text-3xl font-bold text-emerald-400">
          FedMed Dashboard
        </h1>
      </main>
    </div>
  );
}

export default App;