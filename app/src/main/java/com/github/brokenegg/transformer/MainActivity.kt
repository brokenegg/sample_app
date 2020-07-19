package com.github.brokenegg.transformer

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    val MAX_LEN = 10
    private var translator: Translator? = null
    private var editTextInput: EditText? = null
    private var editTextOutput: EditText? = null
    private var spinner: Spinner? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        translator = Translator(this)
        editTextInput = findViewById(R.id.editTextInput)
        editTextOutput = findViewById(R.id.editTextOutput)
        spinner = findViewById(R.id.spinner)
        val button = findViewById<Button>(R.id.buttonTranslate)
        button.setOnClickListener {
            val pos = spinner!!.getSelectedItemPosition()
            val langId: Long = when (pos) {
                0 -> Translator.LANG_ENGLISH
                1 -> Translator.LANG_SPANISH
                2 -> Translator.LANG_JAPANESE
                3 -> Translator.LANG_CHITCHAT
                else -> 0
            }

            val inputText = editTextInput!!.text.toString()
            val outputText = translator!!.run(inputText, langId, MAX_LEN)
            editTextOutput!!.setText(outputText)
        }
    }
}