// src/components/Footer.js
import React from 'react';

function Footer() {
  return (
    <footer className="bg-gray-200 text-center p-4 mt-10">
      <p className="text-sm text-gray-600">
        &copy; {new Date().getFullYear()} Reas ML App. All rights reserved.
      </p>
    </footer>
  );
}

export default Footer;
