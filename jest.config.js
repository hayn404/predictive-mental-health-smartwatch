/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/__tests__'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        // Override strict settings for test compatibility
        strict: true,
        esModuleInterop: true,
        module: 'commonjs',
        moduleResolution: 'node',
        jsx: 'react',
        paths: { '@/*': ['./*'] },
      },
    }],
  },
  testMatch: ['**/__tests__/**/*.test.ts'],
  collectCoverageFrom: [
    'services/ai/**/*.ts',
    '!services/ai/index.ts',
    '!services/ai/db.ts',
    '!services/ai/healthConnect.ts',
    '!services/ai/notifications.ts',
    '!services/ai/dataExport.ts',
    '!services/ai/speechRecognition.ts',
    '!services/ai/audioRecorder.ts',
  ],
};
