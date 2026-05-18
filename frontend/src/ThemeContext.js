import { createContext, useContext } from 'react'

export const ThemeContext = createContext(true) // true = dark by default

export const useTheme = () => useContext(ThemeContext)
