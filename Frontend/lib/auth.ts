import { User } from './mock-data'

const USERS_STORAGE_KEY = 'qace_users'
const CURRENT_USER_KEY = 'qace_current_user'

// ============================================
// Utility Functions
// ============================================

export function isClient(): boolean {
  return typeof window !== 'undefined'
}

// Simple hash function for demo (not secure for production!)
function simpleHash(str: string): string {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash
  }
  return Math.abs(hash).toString(36)
}

// ============================================
// User Storage (All registered users)
// ============================================

interface StoredUser extends User {
  passwordHash: string
  onboardingCompleted: boolean
}

function getAllUsers(): StoredUser[] {
  if (!isClient()) return []
  try {
    const usersStr = localStorage.getItem(USERS_STORAGE_KEY)
    if (!usersStr) return []
    return JSON.parse(usersStr) as StoredUser[]
  } catch {
    return []
  }
}

function saveAllUsers(users: StoredUser[]): void {
  if (!isClient()) return
  try {
    localStorage.setItem(USERS_STORAGE_KEY, JSON.stringify(users))
  } catch (error) {
    console.error('Failed to save users:', error)
  }
}

function findUserByEmail(email: string): StoredUser | null {
  const users = getAllUsers()
  return users.find(u => u.email.toLowerCase() === email.toLowerCase()) || null
}

// ============================================
// Current User (Logged in user)
// ============================================

export function getUser(): User | null {
  if (!isClient()) return null
  try {
    const userStr = localStorage.getItem(CURRENT_USER_KEY)
    if (!userStr) return null
    return JSON.parse(userStr) as User
  } catch {
    return null
  }
}

export function setUser(user: User): void {
  if (!isClient()) return
  try {
    localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user))
    // Also update in users array
    const users = getAllUsers()
    const index = users.findIndex(u => u.id === user.id)
    if (index !== -1) {
      users[index] = { ...users[index], ...user }
      saveAllUsers(users)
    }
  } catch (error) {
    console.error('Failed to save user:', error)
  }
}

export function removeUser(): void {
  if (!isClient()) return
  try {
    localStorage.removeItem(CURRENT_USER_KEY)
  } catch (error) {
    console.error('Failed to remove user:', error)
  }
}

export function isAuthenticated(): boolean {
  if (!isClient()) return false
  return getUser() !== null
}

// ============================================
// Auth Functions
// ============================================

export interface SignupData {
  email: string
  password: string
  name: string
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface AuthResult {
  success: boolean
  user?: User
  error?: string
  isNewUser?: boolean
}

export async function signup(data: SignupData): Promise<AuthResult> {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500))

  // Check if email already exists
  const existingUser = findUserByEmail(data.email)
  if (existingUser) {
    return { success: false, error: 'An account with this email already exists' }
  }

  // Create new user
  const newUser: StoredUser = {
    id: `user-${Date.now()}`,
    email: data.email,
    name: data.name,
    passwordHash: simpleHash(data.password),
    role: '',
    experienceLevel: 'entry',
    interviewTypes: [],
    goals: {},
    createdAt: new Date().toISOString(),
    onboardingCompleted: false,
  }

  // Save to users array
  const users = getAllUsers()
  users.push(newUser)
  saveAllUsers(users)

  // Return success but DON'T log in - user should log in manually
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { passwordHash: _, ...userWithoutPassword } = newUser

  return { success: true, user: userWithoutPassword, isNewUser: true }
}

export async function login(email: string, password: string): Promise<AuthResult> {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500))

  // Find user
  const storedUser = findUserByEmail(email)
  if (!storedUser) {
    return { success: false, error: 'No account found with this email' }
  }

  // Check password
  if (storedUser.passwordHash !== simpleHash(password)) {
    return { success: false, error: 'Incorrect password' }
  }

  // Set as current user (without passwordHash)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { passwordHash: __, ...userWithoutPassword } = storedUser
  localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(userWithoutPassword))

  return { 
    success: true, 
    user: userWithoutPassword,
    isNewUser: !storedUser.onboardingCompleted 
  }
}

export function logout(): void {
  removeUser()
}

// ============================================
// Onboarding
// ============================================

export function hasCompletedOnboarding(): boolean {
  const user = getUser()
  if (!user) return false
  const storedUser = findUserByEmail(user.email)
  return storedUser?.onboardingCompleted ?? false
}

export function completeOnboarding(profileData: Partial<User>): void {
  const currentUser = getUser()
  if (!currentUser) return

  const users = getAllUsers()
  const index = users.findIndex(u => u.id === currentUser.id)
  
  if (index !== -1) {
    users[index] = {
      ...users[index],
      ...profileData,
      onboardingCompleted: true,
    }
    saveAllUsers(users)

    // Update current user
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { passwordHash: ___, ...userWithoutPassword } = users[index]
    localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(userWithoutPassword))
  }
}

// ============================================
// Legacy support (for backwards compatibility)
// ============================================

export { getUser as getCurrentUser }

