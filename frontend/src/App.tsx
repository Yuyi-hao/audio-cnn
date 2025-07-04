import { Routes, Route } from 'react-router-dom'
import './App.css'
import HomePage from './pages/home/HomePage'
import NotFoundPage from './pages/notFound/NotFoundPage'

function App() {
  return (
    <>
    <Routes>
      <Route path='/' element={<HomePage/>} />
      <Route path='/*' element={<NotFoundPage/>} />
    </Routes>
    </>
  )
}

export default App
