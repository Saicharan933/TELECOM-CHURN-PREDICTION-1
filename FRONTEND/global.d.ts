declare module 'react' {
  export type ReactNode = any;
  export interface HTMLAttributes<T> { [key: string]: any }
  export interface ButtonHTMLAttributes<T> extends HTMLAttributes<T> {}
  export interface InputHTMLAttributes<T> extends HTMLAttributes<T> {}
  export interface SelectHTMLAttributes<T> extends HTMLAttributes<T> {}
  export interface TableHTMLAttributes<T> extends HTMLAttributes<T> {}
  export interface ThHTMLAttributes<T> extends HTMLAttributes<T> {}
  export interface TdHTMLAttributes<T> extends HTMLAttributes<T> {}
  export function useState<T = any>(initial?: any): [T, (v: any) => void];
  export function useEffect(fn: () => void | (() => void), deps?: any[]): void;
  export const Fragment: any;
  const ReactDefaultExport: any;
  export default ReactDefaultExport;
}

declare module 'react/jsx-runtime' {
  export function jsx(type: any, props?: any): any;
  export function jsxs(type: any, props?: any): any;
  export function jsxDEV(type: any, props?: any): any;
}

declare module 'next/link';
declare module 'next/navigation';
declare module 'recharts';
declare module 'lucide-react';
declare module '@/lib/utils';

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any
  }
}

declare namespace React {
  type ReactNode = any;
  interface HTMLAttributes<T = any> { [key: string]: any }
  interface ButtonHTMLAttributes<T = any> extends HTMLAttributes<T> {}
  interface InputHTMLAttributes<T = any> extends HTMLAttributes<T> {}
  interface SelectHTMLAttributes<T = any> extends HTMLAttributes<T> {}
  interface TableHTMLAttributes<T = any> extends HTMLAttributes<T> {}
  interface ThHTMLAttributes<T = any> extends HTMLAttributes<T> {}
  interface TdHTMLAttributes<T = any> extends HTMLAttributes<T> {}
  interface TdHTMLAttributes<T = any> extends HTMLAttributes<T> {}
  type ComponentType<T = any> = any;
}
