import { useState, useRef, useEffect, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Check } from 'lucide-react';

/**
 * Custom glass-themed dropdown.
 * Uses a portal so the menu escapes parent stacking contexts
 * (backdrop-filter / transforms / overflow).
 *
 * options: [{ value, label }]
 */
const Dropdown = ({ value, onChange, options, placeholder = 'Select', className = '' }) => {
  const [open, setOpen] = useState(false);
  const [menuPos, setMenuPos] = useState({ top: 0, left: 0, width: 0 });
  const triggerRef = useRef(null);
  const menuRef = useRef(null);

  // Position menu under trigger
  const updatePosition = () => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setMenuPos({
      top: rect.bottom + 8,
      left: rect.left,
      width: rect.width,
    });
  };

  useLayoutEffect(() => {
    if (!open) return;
    updatePosition();
  }, [open]);

  // Reposition on scroll / resize while open
  useEffect(() => {
    if (!open) return;
    const handle = () => updatePosition();
    window.addEventListener('scroll', handle, true);
    window.addEventListener('resize', handle);
    return () => {
      window.removeEventListener('scroll', handle, true);
      window.removeEventListener('resize', handle);
    };
  }, [open]);

  // Outside click
  useEffect(() => {
    if (!open) return;
    const handleClick = (e) => {
      if (triggerRef.current?.contains(e.target)) return;
      if (menuRef.current?.contains(e.target))   return;
      setOpen(false);
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  // Escape
  useEffect(() => {
    if (!open) return;
    const handleKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open]);

  const selected = options.find(opt => opt.value === value);

  return (
    <>
      <div className={`relative ${className}`}>
        <button
          ref={triggerRef}
          type="button"
          onClick={() => setOpen(!open)}
          className={`w-full flex items-center justify-between gap-2 px-3 py-2 text-sm
                      bg-white rounded-xl border transition-all duration-200
                      text-neutral-black cursor-pointer outline-none
                      ${open
                        ? 'border-brand-400 ring-2 ring-brand-300/40 shadow-sm'
                        : 'border-neutral-border hover:border-brand-300'}`}
        >
          <span className="truncate font-medium">
            {selected?.label || placeholder}
          </span>
          <ChevronDown
            className={`w-4 h-4 text-neutral-muted shrink-0 transition-transform duration-200 ${
              open ? 'rotate-180 text-brand-500' : ''
            }`}
          />
        </button>
      </div>

      {open && createPortal(
        <div
          ref={menuRef}
          className="fixed z-[9999] glass-solid rounded-xl shadow-glass-lg
                     overflow-hidden animate-slide-down max-h-72 overflow-y-auto"
          style={{
            top:   `${menuPos.top}px`,
            left:  `${menuPos.left}px`,
            width: `${menuPos.width}px`,
            minWidth: '160px',
          }}
        >
          <div className="py-1">
            {options.map((opt) => {
              const isActive = opt.value === value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => {
                    onChange(opt.value);
                    setOpen(false);
                  }}
                  className={`w-full flex items-center justify-between gap-2 px-3 py-2 text-sm
                              text-left transition-colors duration-150
                              ${isActive
                                ? 'bg-brand-50 text-brand-600 font-semibold'
                                : 'text-neutral-black hover:bg-brand-50/60 hover:text-brand-600'}`}
                >
                  <span className="truncate capitalize">{opt.label}</span>
                  {isActive && <Check className="w-3.5 h-3.5 shrink-0 text-brand-500" />}
                </button>
              );
            })}
          </div>
        </div>,
        document.body
      )}
    </>
  );
};

export default Dropdown;
